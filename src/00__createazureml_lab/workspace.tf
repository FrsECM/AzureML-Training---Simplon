# Dependent resources for Azure Machine Learning

##### Define Application Insight
# https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/application_insights
resource "azurerm_application_insights" "default" {
  name                = "ai${var.name}${var.michelin_id}${var.week}"
  location            = azurerm_resource_group.default.location
  resource_group_name = azurerm_resource_group.default.name
  application_type    = "web"
}

##### Define KeyVault
# https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/key_vault
resource "azurerm_key_vault" "default" {
  name                     = "kv${var.name}${var.michelin_id}${var.week}"
  location                 = azurerm_resource_group.default.location
  resource_group_name      = azurerm_resource_group.default.name
  tenant_id                = data.azurerm_client_config.current.tenant_id
  sku_name                 = "standard"
  purge_protection_enabled = false
}

##### Define storage account
# https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/storage_account
resource "azurerm_storage_account" "default" {
  name                            = "st${var.name}${var.michelin_id}${var.week}aml"
  location                        = azurerm_resource_group.default.location
  resource_group_name             = azurerm_resource_group.default.name
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  min_tls_version                 = "TLS1_2"
}

resource "azurerm_storage_account" "data" {
  name                            = "st${var.name}${var.michelin_id}${var.week}data"
  location                        = azurerm_resource_group.default.location
  resource_group_name             = azurerm_resource_group.default.name
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  min_tls_version                 = "TLS1_2"
}


##### Define ACR
# https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/container_registry

resource "azurerm_container_registry" "default" {
  name                = "cr${var.name}${var.michelin_id}${var.week}"
  location            = azurerm_resource_group.default.location
  resource_group_name = azurerm_resource_group.default.name
  sku                 = "Standard"
  admin_enabled       = true
}

# Machine Learning workspace
resource "azurerm_machine_learning_workspace" "default" {
  name                    = "mlw${var.name}${var.michelin_id}${var.week}"
  location                = azurerm_resource_group.default.location
  resource_group_name     = azurerm_resource_group.default.name
  application_insights_id = azurerm_application_insights.default.id
  key_vault_id            = azurerm_key_vault.default.id
  storage_account_id      = azurerm_storage_account.default.id
  container_registry_id   = azurerm_container_registry.default.id

  identity {
    type = "SystemAssigned"
  }
}

# Machine Learning compute instance
resource "azurerm_machine_learning_compute_instance" "default" {
  name                          = "cpu-${var.michelin_id}-ci"
  location                      = azurerm_resource_group.default.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.default.id
  virtual_machine_size          = "STANDARD_DS11_V2"
  authorization_type            = "personal"

  description        = "Compute Instance for dev"

}