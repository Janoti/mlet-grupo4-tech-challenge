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
#   ./infra/deploy.sh                    # deploy completo
#   ./infra/deploy.sh --stack-only       # so atualiza a stack (sem rebuild)
#   ./infra/deploy.sh --build-only       # so build + push (sem deploy CFN)
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
IMAGE_TAG="${IMAGE_TAG:-latest}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
KEY_PAIR_NAME="${KEY_PAIR_NAME:-}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-mletg4}"

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

    # Tag + Push
    docker tag "$STACK_NAME/churn-api:$IMAGE_TAG" "$ECR_URI:$IMAGE_TAG"
    docker push "$ECR_URI:$IMAGE_TAG"

    log "Imagem publicada: $ECR_URI:$IMAGE_TAG"
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
main() {
    check_prerequisites

    case "${1:-}" in
        --build-only)
            build_and_push
            ;;
        --stack-only)
            deploy_stack
            ;;
        --destroy)
            destroy_stack
            ;;
        *)
            build_and_push
            deploy_stack
            ;;
    esac
}

main "$@"
