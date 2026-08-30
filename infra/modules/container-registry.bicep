param name string
param location string
param tags object = {}

resource acr 'Microsoft.ContainerRegistry/registries@2025-04-01' = {
  name: name
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
    anonymousPullEnabled: false
  }
  tags: tags
}

output name string = acr.name
output id string = acr.id
output loginServer string = acr.properties.loginServer
