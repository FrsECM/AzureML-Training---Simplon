variable "name" {
  type        = string
  description = "Name of the AzureML workspace"
  default     = "amlfdme"
}

variable "michelin_id" {
  type        = string
  description = "Suffix to add to the name that is user specific"
  default     = "279814" # Edit with your Michelin ID  (without F or E, only digits)
}
variable "week" {
  type        = string
  description = "Week"
  default     = "2248" # Edit with your Michelin ID and week
}
variable "location" {
  type        = string
  description = "Location of the resources"
  default     = "westeurope"
}

variable "resource_group" {
  type        = string
  description = "Resource Group where to deploy"
  default     = "F279814-Guillaume_Ramelet" # Edit with your Lab Space
}

variable "subscription_id" {
  type        = string
  description = "Target Subscription"
  default = "049118e2-4814-401b-a34d-a67a35abc5a9"
}
