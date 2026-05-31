from traceguard.utils.config import load_config


def test_loads_default_config():
    config = load_config()

    assert config["dataset"]["name"] == "cifar10"
    assert config["defense"]["name"] == "fedavg"
    assert config["attack"]["name"] == "none"


def test_debug_config_overrides_default():
    config = load_config(debug=True)

    assert config["debug"]["enabled"] is True
    assert config["training"]["rounds"] == 1
    assert config["federated"]["num_clients"] == 3


def test_cli_dataset_override_wins():
    config = load_config(cli_overrides={"dataset.name": "cifar100"})

    assert config["dataset"]["name"] == "cifar100"


def test_cli_defense_override_wins():
    config = load_config(cli_overrides={"defense.name": "traceguard"})

    assert config["defense"]["name"] == "traceguard"
