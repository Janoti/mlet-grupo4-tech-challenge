"""Testes para helpers MLflow."""


from churn_prediction.mlflow_utils import (
    build_standard_tags,
    compute_dataset_version,
    flatten_params,
    format_note_content,
    get_author,
    get_git_sha,
)


class TestGitSha:
    def test_returns_string(self):
        sha = get_git_sha(short=True)
        assert isinstance(sha, str)
        assert len(sha) > 0

    def test_short_sha_is_shorter(self):
        short = get_git_sha(short=True)
        full = get_git_sha(short=False)
        if short != "unknown" and full != "unknown":
            assert len(short) <= len(full)


class TestAuthor:
    def test_returns_string(self):
        author = get_author()
        assert isinstance(author, str)
        assert len(author) > 0


class TestDatasetVersion:
    def test_returns_hash_for_existing_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n")
        h = compute_dataset_version(f)
        assert len(h) == 12
        assert h != "missing"

    def test_returns_missing_for_nonexistent(self):
        assert compute_dataset_version("/tmp/nonexistent_xyz_12345.csv") == "missing"

    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("same content\n")
        f2.write_text("same content\n")
        assert compute_dataset_version(f1) == compute_dataset_version(f2)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("content one\n")
        f2.write_text("content two\n")
        assert compute_dataset_version(f1) != compute_dataset_version(f2)


class TestBuildStandardTags:
    def test_minimal_tags(self):
        tags = build_standard_tags(phase="baseline")
        assert "git_sha" in tags
        assert "author" in tags
        assert tags["run_phase"] == "baseline"

    def test_with_dataset(self, tmp_path):
        f = tmp_path / "x.csv"
        f.write_text("1\n")
        tags = build_standard_tags(phase="mlp", dataset_path=f)
        assert "dataset_version" in tags
        assert tags["dataset_version"] != "missing"

    def test_with_model_type(self):
        tags = build_standard_tags(phase="baseline", model_type="log_reg")
        assert tags["model_type"] == "log_reg"

    def test_with_extra_tags(self):
        tags = build_standard_tags(
            phase="tuning", extra={"search_strategy": "bayesian"}
        )
        assert tags["search_strategy"] == "bayesian"


class TestFlattenParams:
    def test_single_level(self):
        assert flatten_params({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested(self):
        result = flatten_params({"model": {"hidden1": 128, "hidden2": 64}})
        assert result == {"model.hidden1": 128, "model.hidden2": 64}

    def test_deep_nesting(self):
        result = flatten_params({"a": {"b": {"c": 42}}})
        assert result == {"a.b.c": 42}

    def test_mixed(self):
        result = flatten_params({
            "model": {"hidden1": 128},
            "train": {"lr": 0.001, "batch_size": 512},
            "data": {"split": 0.8},
        })
        assert result == {
            "model.hidden1": 128,
            "train.lr": 0.001,
            "train.batch_size": 512,
            "data.split": 0.8,
        }

    def test_custom_separator(self):
        result = flatten_params({"a": {"b": 1}}, separator="/")
        assert result == {"a/b": 1}


class TestFormatNoteContent:
    def test_minimal(self):
        note = format_note_content(
            objective="Testar baseline",
            approach="LogReg com C=0.1",
        )
        assert "### Objetivo" in note
        assert "### Abordagem" in note
        assert "Testar baseline" in note

    def test_with_all_sections(self):
        note = format_note_content(
            objective="obj",
            approach="appr",
            dataset_info="50k linhas",
            expected_outcome="AUC > 0.85",
            caveats="dataset sintético",
        )
        assert "### Objetivo" in note
        assert "### Abordagem" in note
        assert "### Dataset" in note
        assert "### Resultado esperado" in note
        assert "### Limitações" in note

    def test_optional_sections_omitted(self):
        note = format_note_content(objective="o", approach="a")
        assert "### Dataset" not in note
        assert "### Resultado esperado" not in note
        assert "### Limitações" not in note
