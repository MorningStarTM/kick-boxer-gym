from pathlib import Path

# Files that usually already exist in a GitHub repo
GITHUB_DEFAULT_FILES = {"README.md", ".gitignore", "LICENSE"}

PROJECT_STRUCTURE = {
    "src": {
        "kick_boxer_gym": {
            "__init__.py": None,
            "envs": {
                "__init__.py": None,
                "simple_human_bullet_env.py": None,
                "combat_1v1_bullet_env.py": None,
                "reward.py": None,
                "wrappers.py": None,
            },
            "assets": {
                "urdf": {
                    "simple_human.urdf": None,
                    "plane.urdf": None,
                    "meshes": {}
                },
                "textures": {}
            },
            "sim": {
                "__init__.py": None,
                "bullet_client.py": None,
                "humanoid_loader.py": None,
                "control.py": None,
                "contacts.py": None,
            },
            "utils": {
                "__init__.py": None,
                "seeding.py": None,
                "math.py": None,
                "logging.py": None,
            },
            "configs": {
                "env.yaml": None,
                "train.yaml": None,
            },
        }
    },
    "tools": {
        "pb_spawn_test.py": None,
        "print_joint_map.py": None,
        "debug_contacts.py": None,
    },
    "training": {
        "train_ppo.py": None,
        "self_play.py": None,
        "callbacks.py": None,
    },
    "eval": {
        "winrate_eval.py": None,
        "record_episode.py": None,
    },
    "tests": {
        "test_env_rollout.py": None,
        "test_reset_step_api.py": None,
    },
    "docs": {
        "spec.md": None,
        "design.md": None,
    },
    "requirements.txt": None,
    "pyproject.toml": None,
}

def create_structure(base_path: Path, structure: dict):
    for name, content in structure.items():
        path = base_path / name

        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)
        else:
            # If GitHub already created these, don't touch them
            if name in GITHUB_DEFAULT_FILES and path.exists():
                print(f"Skip existing GitHub file: {path}")
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                print(f"Create file: {path}")
                path.touch()
            else:
                print(f"Exists: {path}")

if __name__ == "__main__":
    repo_root = Path.cwd()   # ✅ current folder (already inside kick-boxer-gym)
    create_structure(repo_root, PROJECT_STRUCTURE)
    print("\n✅ Done: project structure created in current repo folder.")
