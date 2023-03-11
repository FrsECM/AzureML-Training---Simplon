terraform {
  required_version = ">=1.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=2.76.0"
    }
  }
}
# Largement inspiré de :
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-terraform?tabs=publicworkspace


provider "azurerm" {
  subscription_id = var.subscription_id
  features {
    key_vault {
      purge_soft_delete_on_destroy = true
    }
  }
}

data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "default" {
  name     = var.resource_group
  location = var.location
}
