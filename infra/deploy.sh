#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Deploy end-to-end: build imagem + push ECR + deploy CloudFormation
#
# Adequado às políticas de governança da conta hubfintech/fintechmagalu:
#   - Tags obrigatórias (se_org, se_environment, se_resource, se_pci_machine, se_conta_pci)
#   - KMS encryption (EBS, ECR, CloudWatch Logs)
#   - IMDSv2 obrigatório
#   - VPC Flow Logs habilitados
#   - Security Groups sem SSH aberto para 0.0.0.0/0
#
# Uso:
#   ./infra/deploy.sh                    # deploy completo (treino + build + deploy)
#   ./infra/deploy.sh --skip-train       # build + deploy (sem retreinar)
#   ./infra/deploy.sh --train-only       # so treina e exporta modelo (registra no MLflow prod)
#   ./infra/deploy.sh --build-only       # so build + push (sem deploy CFN)
#   ./infra/deploy.sh --stack-only       # so atualiza a stack (sem rebuild)
#   ./infra/deploy.sh --destroy          # destroi stack + ECR images
#
# Pre-requisitos:
#   - AWS CLI v2 configurado (aws sts get-caller-identity)
#   - Docker rodando
#   - Modelo exportado em models/churn_pipeline.joblib
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuracoes (edite conforme necessario)
# ---------------------------------------------------------------------------
STACK_NAME="${STACK_NAME:-churn-prediction-stack}"
AWS_REGION="${AWS_REGION:-sa-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
KEY_PAIR_NAME="${KEY_PAIR_NAME:-}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-mletg4}"

# Versionamento automático: git-hash + timestamp (ex: abc1234-20260502-1812)
# Pode ser sobrescrito com IMAGE_TAG=v1.0.0 ./infra/deploy.sh
GIT_SHORT_HASH=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
AUTO_TAG="${GIT_SHORT_HASH}-$(date +%Y%m%d-%H%M)"
IMAGE_TAG="${IMAGE_TAG:-$AUTO_TAG}"

# Tags de governança obrigatórias (padrão hubfintech)
TAG_SE_ORG="${TAG_SE_ORG:-fintech}"
TAG_SE_ENVIRONMENT="${TAG_SE_ENVIRONMENT:-sso-hml}"
TAG_SE_RESOURCE="${TAG_SE_RESOURCE:-ec2}"
TAG_SE_PCI_MACHINE="${TAG_SE_PCI_MACHINE:-no}"
TAG_SE_CONTA_PCI="${TAG_SE_CONTA_PCI:-no}"

# Caminhos
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEMPLATE_FILE="$SCRIPT_DIR/template.yaml"
DOCKERFILE="$SCRIPT_DIR/Dockerfile.prod"

# ---------------------------------------------------------------------------
# Funcoes utilitarias
# ---------------------------------------------------------------------------
log()   { echo -e "\033[1;32m[deploy]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[warn]\033[0m $*"; }
error() { echo -e "\033[1;31m[error]\033[0m $*" >&2; exit 1; }

check_prerequisites() {
    log "Verificando pre-requisitos..."

    command -v aws >/dev/null 2>&1 || error "AWS CLI nao encontrado. Instale: https://aws.amazon.com/cli/"
    command -v docker >/dev/null 2>&1 || error "Docker nao encontrado."

    # Verifica credenciais AWS
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) \
        || error "Credenciais AWS invalidas. Execute 'aws configure' ou renove seu SSO."

    log "AWS Account: $AWS_ACCOUNT_ID | Region: $AWS_REGION"

    # Verifica se o modelo existe
    if [ ! -f "$PROJECT_ROOT/models/churn_pipeline.joblib" ]; then
        error "Modelo nao encontrado em models/churn_pipeline.joblib. Execute:\n  PYTHONPATH=src poetry run python scripts/export_model.py"
    fi

    ECR_REPO_NAME="$STACK_NAME/churn-api"
    ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"
}

# ---------------------------------------------------------------------------
# Treino + Export do modelo (registra no MLflow de produção)
# ---------------------------------------------------------------------------
train_and_export() {
    log "=== FASE 0: Treino e exportação do modelo ==="

    # Resolve a URL do MLflow de produção
    local MLFLOW_URL
    MLFLOW_URL=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='MLflowUrl'].OutputValue" \
        --output text 2>/dev/null || echo "")

    if [ -n "$MLFLOW_URL" ] && [ "$MLFLOW_URL" != "None" ]; then
        log "MLflow de produção detectado: $MLFLOW_URL"
        log "Executando treino remoto na EC2 (system metrics + artifacts automáticos)..."
        train_remote "$MLFLOW_URL"
    else
        log "MLflow de produção não encontrado. Treinando localmente..."
        train_local
    fi
}

train_local() {
    # Gerar dados se não existem
    if [ ! -f "$PROJECT_ROOT/data/raw/telecom_churn_base_extended.csv" ]; then
        log "Gerando dataset sintético..."
        (cd "$PROJECT_ROOT" && poetry run python scripts/generate_synthetic.py \
            --n-rows 50000 --seed 42 --out-dir data/raw)
    fi

    log "Executando pipeline de treino (notebooks)..."
    (cd "$PROJECT_ROOT" && make notebooks)

    log "Selecionando e exportando champion..."
    (cd "$PROJECT_ROOT" && PYTHONPATH=src poetry run python scripts/export_model.py)

    [ -f "$PROJECT_ROOT/models/churn_pipeline.joblib" ] \
        || error "Modelo não foi exportado."
    log "Modelo exportado: models/churn_pipeline.joblib"
}

train_remote() {
    local MLFLOW_URL="$1"

    log "Build da imagem de treino..."
    # Troca dockerignore temporariamente (treino precisa de data/raw e notebooks)
    mv "$PROJECT_ROOT/.dockerignore" "$PROJECT_ROOT/.dockerignore.prod"
    cp "$SCRIPT_DIR/.dockerignore.train" "$PROJECT_ROOT/.dockerignore"
    docker build \
        -f "$SCRIPT_DIR/Dockerfile.train" \
        -t "$STACK_NAME/churn-train:latest" \
        "$PROJECT_ROOT"
    mv "$PROJECT_ROOT/.dockerignore.prod" "$PROJECT_ROOT/.dockerignore"

    log "Push da imagem de treino para ECR..."
    local TRAIN_REPO="$STACK_NAME/churn-train"
    local TRAIN_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$TRAIN_REPO"

    aws ecr create-repository \
        --repository-name "$TRAIN_REPO" \
        --region "$AWS_REGION" >/dev/null 2>&1 || true

    docker tag "$STACK_NAME/churn-train:latest" "$TRAIN_URI:latest"
    docker push "$TRAIN_URI:latest"

    log "Executando treino na EC2..."
    local INSTANCE_ID
    INSTANCE_ID=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" \
        --output text)

    local CMD_ID
    CMD_ID=$(aws ssm send-command \
        --document-name "AWS-RunShellScript" \
        --instance-ids "$INSTANCE_ID" \
        --parameters "{\"commands\":[
            \"aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com 2>&1\",
            \"docker pull $TRAIN_URI:latest 2>&1 | tail -3\",
            \"docker run --rm --network churn-app_churn-net -e MLFLOW_TRACKING_URI=http://mlflow:5000 -v /opt/churn-app/models:/app/models $TRAIN_URI:latest 2>&1\",
            \"ls -la /opt/churn-app/models/churn_pipeline.joblib\"
        ]}" \
        --timeout-seconds 900 \
        --region "$AWS_REGION" \
        --query 'Command.CommandId' --output text)

    log "Treino iniciado (command_id=$CMD_ID). Aguardando conclusão (~5-10 min)..."

    # Aguardar conclusão
    local STATUS="InProgress"
    while [ "$STATUS" = "InProgress" ] || [ "$STATUS" = "Pending" ]; do
        sleep 30
        STATUS=$(aws ssm get-command-invocation \
            --command-id "$CMD_ID" \
            --instance-id "$INSTANCE_ID" \
            --region "$AWS_REGION" \
            --query 'Status' --output text 2>/dev/null || echo "InProgress")
        log "  Status: $STATUS"
    done

    if [ "$STATUS" != "Success" ]; then
        warn "Treino remoto falhou (status=$STATUS). Logs:"
        aws ssm get-command-invocation \
            --command-id "$CMD_ID" \
            --instance-id "$INSTANCE_ID" \
            --region "$AWS_REGION" \
            --query 'StandardErrorContent' --output text 2>/dev/null | tail -20
        error "Treino remoto falhou. Verifique os logs acima."
    fi

    log "Treino remoto concluído. Copiando modelo da EC2..."

    # Copiar modelo da EC2 para local (para o build da imagem de produção)
    CMD_ID=$(aws ssm send-command \
        --document-name "AWS-RunShellScript" \
        --instance-ids "$INSTANCE_ID" \
        --parameters '{"commands":["cat /opt/churn-app/models/churn_pipeline.joblib | base64"]}' \
        --region "$AWS_REGION" \
        --query 'Command.CommandId' --output text)
    sleep 10

    aws ssm get-command-invocation \
        --command-id "$CMD_ID" \
        --instance-id "$INSTANCE_ID" \
        --region "$AWS_REGION" \
        --query 'StandardOutputContent' --output text \
        | base64 -d > "$PROJECT_ROOT/models/churn_pipeline.joblib"

    [ -s "$PROJECT_ROOT/models/churn_pipeline.joblib" ] \
        || error "Falha ao copiar modelo da EC2."
    log "Modelo copiado: models/churn_pipeline.joblib"
}

# ---------------------------------------------------------------------------
# Build + Push da imagem Docker
# ---------------------------------------------------------------------------
build_and_push() {
    log "=== FASE 1: Build da imagem Docker otimizada ==="

    docker build \
        -f "$DOCKERFILE" \
        -t "$STACK_NAME/churn-api:$IMAGE_TAG" \
        "$PROJECT_ROOT"

    IMAGE_SIZE=$(docker images "$STACK_NAME/churn-api:$IMAGE_TAG" --format "{{.Size}}")
    log "Imagem construida: $STACK_NAME/churn-api:$IMAGE_TAG ($IMAGE_SIZE)"

    log "=== FASE 2: Push para ECR ==="

    # Garante que o repositorio ECR existe (cria se necessario)
    if ! aws ecr describe-repositories \
        --repository-names "$ECR_REPO_NAME" \
        --region "$AWS_REGION" >/dev/null 2>&1; then

        aws ecr create-repository \
            --repository-name "$ECR_REPO_NAME" \
            --region "$AWS_REGION" \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=KMS \
            --tags \
                "Key=se_org,Value=$TAG_SE_ORG" \
                "Key=se_environment,Value=$TAG_SE_ENVIRONMENT" \
                "Key=se_resource,Value=ecr" \
                "Key=se_pci_machine,Value=$TAG_SE_PCI_MACHINE" \
                "Key=se_conta_pci,Value=$TAG_SE_CONTA_PCI" \
                "Key=Name,Value=$STACK_NAME-ecr" \
            >/dev/null

        aws ecr put-lifecycle-policy \
            --repository-name "$ECR_REPO_NAME" \
            --region "$AWS_REGION" \
            --lifecycle-policy-text '{"rules":[{"rulePriority":1,"description":"Manter 5 imagens","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":5},"action":{"type":"expire"}}]}' \
            >/dev/null
    fi

    # Login no ECR
    aws ecr get-login-password --region "$AWS_REGION" \
        | docker login --username AWS --password-stdin \
          "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

    # Tag + Push (versão + latest)
    docker tag "$STACK_NAME/churn-api:$IMAGE_TAG" "$ECR_URI:$IMAGE_TAG"
    docker tag "$STACK_NAME/churn-api:$IMAGE_TAG" "$ECR_URI:latest"
    docker push "$ECR_URI:$IMAGE_TAG"
    docker push "$ECR_URI:latest"

    log "Imagem publicada: $ECR_URI:$IMAGE_TAG (+ latest)"
}

# ---------------------------------------------------------------------------
# Deploy da stack CloudFormation
# ---------------------------------------------------------------------------
deploy_stack() {
    log "=== FASE 3: Deploy CloudFormation ==="

    # Valida template
    aws cloudformation validate-template \
        --template-body "file://$TEMPLATE_FILE" \
        --region "$AWS_REGION" >/dev/null \
    || error "Template CloudFormation invalido."

    log "Template validado. Iniciando deploy da stack '$STACK_NAME'..."

    aws cloudformation deploy \
        --template-file "$TEMPLATE_FILE" \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --capabilities CAPABILITY_NAMED_IAM \
        --parameter-overrides \
            InstanceType="$INSTANCE_TYPE" \
            KeyPairName="$KEY_PAIR_NAME" \
            ImageTag="$IMAGE_TAG" \
            GrafanaPassword="$GRAFANA_PASSWORD" \
            SeOrg="$TAG_SE_ORG" \
            SeEnvironment="$TAG_SE_ENVIRONMENT" \
            SePciMachine="$TAG_SE_PCI_MACHINE" \
            SeContaPci="$TAG_SE_CONTA_PCI" \
        --tags \
            "se_org=$TAG_SE_ORG" \
            "se_environment=$TAG_SE_ENVIRONMENT" \
            "se_resource=cloudformation" \
            "se_pci_machine=$TAG_SE_PCI_MACHINE" \
            "se_conta_pci=$TAG_SE_CONTA_PCI" \
            "Name=$STACK_NAME" \
        --no-fail-on-empty-changeset

    log "Stack deployada. Aguardando outputs..."

    # Exibe outputs
    echo ""
    echo "============================================================"
    echo "  DEPLOY CONCLUIDO"
    echo "============================================================"

    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --query "Stacks[0].Outputs[*].[OutputKey,OutputValue]" \
        --output table

    echo ""

    API_URL=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
        --output text)

    log "Aguarde ~3-5 min para a EC2 inicializar os containers."
    log "Teste com:"
    echo "  curl $API_URL/health"
    echo "  curl -X POST $API_URL/predict -H 'Content-Type: application/json' -d '{\"age\":45,\"gender\":\"female\",\"plan_type\":\"pos\",\"monthly_charges\":120,\"nps_score\":3}'"
}

# ---------------------------------------------------------------------------
# Destroy
# ---------------------------------------------------------------------------
cleanup_guardduty_endpoints() {
    # GuardDuty cria VPC Endpoints automaticamente em VPCs novas.
    # Esses endpoints impedem a deleção da subnet/VPC pelo CloudFormation.
    local vpc_id="$1"
    [ -z "$vpc_id" ] && return 0

    log "Verificando VPC Endpoints do GuardDuty na VPC $vpc_id..."
    local endpoints
    endpoints=$(aws ec2 describe-vpc-endpoints \
        --filters "Name=vpc-id,Values=$vpc_id" "Name=tag:GuardDutyManaged,Values=true" \
        --query 'VpcEndpoints[*].VpcEndpointId' \
        --output text \
        --region "$AWS_REGION" 2>/dev/null || true)

    if [ -n "$endpoints" ] && [ "$endpoints" != "None" ]; then
        log "Deletando VPC Endpoints do GuardDuty: $endpoints"
        aws ec2 delete-vpc-endpoints \
            --vpc-endpoint-ids $endpoints \
            --region "$AWS_REGION" >/dev/null 2>&1 || true
        # Aguarda ENIs serem liberadas
        log "Aguardando liberação das ENIs (20s)..."
        sleep 20

        # Limpa SGs do GuardDuty
        local gd_sgs
        gd_sgs=$(aws ec2 describe-security-groups \
            --filters "Name=vpc-id,Values=$vpc_id" "Name=group-name,Values=GuardDutyManaged*" \
            --query 'SecurityGroups[*].GroupId' \
            --output text \
            --region "$AWS_REGION" 2>/dev/null || true)
        if [ -n "$gd_sgs" ] && [ "$gd_sgs" != "None" ]; then
            for sg in $gd_sgs; do
                aws ec2 delete-security-group --group-id "$sg" --region "$AWS_REGION" 2>/dev/null || true
            done
        fi
    fi
}

destroy_stack() {
    warn "Destruindo stack '$STACK_NAME'..."
    read -p "Tem certeza? (y/N) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 0

    # Identifica a VPC da stack para limpar VPC Endpoints do GuardDuty
    local vpc_id
    vpc_id=$(aws ec2 describe-vpcs \
        --filters "Name=tag:aws:cloudformation:stack-name,Values=$STACK_NAME" \
        --query 'Vpcs[0].VpcId' \
        --output text \
        --region "$AWS_REGION" 2>/dev/null || true)
    [ "$vpc_id" = "None" ] && vpc_id=""

    # Limpa VPC Endpoints do GuardDuty antes de deletar a stack
    cleanup_guardduty_endpoints "$vpc_id"

    # Esvazia ECR antes de deletar (CFN nao deleta repo com imagens)
    log "Removendo imagens do ECR..."
    IMAGES=$(aws ecr list-images \
        --repository-name "$STACK_NAME/churn-api" \
        --query 'imageIds[*]' \
        --output json \
        --region "$AWS_REGION" 2>/dev/null || echo "[]")

    if [ "$IMAGES" != "[]" ] && [ -n "$IMAGES" ]; then
        aws ecr batch-delete-image \
            --repository-name "$STACK_NAME/churn-api" \
            --image-ids "$IMAGES" \
            --region "$AWS_REGION" 2>/dev/null || true
    fi

    log "Deletando stack CloudFormation..."
    aws cloudformation delete-stack \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION"

    aws cloudformation wait stack-delete-complete \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION"

    log "Stack '$STACK_NAME' destruida com sucesso."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
is_stack_deployed() {
    local status
    status=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].StackStatus' \
        --output text 2>/dev/null || echo "NOT_FOUND")
    [[ "$status" == *"COMPLETE"* ]]
}

main() {
    check_prerequisites

    case "${1:-}" in
        --build-only)
            build_and_push
            ;;
        --stack-only)
            deploy_stack
            ;;
        --train-only)
            train_and_export
            ;;
        --destroy)
            destroy_stack
            ;;
        --skip-train)
            build_and_push
            deploy_stack
            ;;
        *)
            if is_stack_deployed; then
                # Stack existe: treino remoto → build → deploy
                log "Stack existente detectada. Fluxo: treino remoto → build → deploy"
                train_and_export
                build_and_push
                deploy_stack
            else
                # Primeiro deploy: build com modelo local → deploy infra → treino remoto → rebuild → redeploy
                log "Primeiro deploy detectado. Fluxo: infra → treino remoto → redeploy"

                # Fase 1: treino local + build + deploy (sobe a infra com modelo inicial)
                train_local
                build_and_push
                deploy_stack

                # Fase 2: aguardar containers e retreinar na EC2 (popula MLflow)
                log "Aguardando containers inicializarem (60s)..."
                sleep 60

                log "=== FASE EXTRA: Retreino remoto para popular MLflow ==="
                train_remote "$(aws cloudformation describe-stacks \
                    --stack-name "$STACK_NAME" \
                    --region "$AWS_REGION" \
                    --query "Stacks[0].Outputs[?OutputKey=='MLflowUrl'].OutputValue" \
                    --output text)"

                # Fase 3: rebuild com modelo do treino remoto + redeploy
                build_and_push
                deploy_stack

                log "Deploy completo! MLflow populado com experimentos e artifacts."
            fi
            ;;
    esac
}

main "$@"
