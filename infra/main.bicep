targetScope = 'resourceGroup'

@description('Azure region for all playground resources. Apps already run in eastus2.')
param location string = 'eastus2'

param acrName string = 'acrhwaiplaye2'
param logAnalyticsName string = 'law-handwriting-ai-e2'
param environmentName string = 'cae-handwriting-ai-playground'
param backendAppName string = 'ca-backend-playground'
param frontendAppName string = 'ca-frontend-playground'
param deployApps bool = false
param backendImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
param frontendImage string = 'mcr.microsoft.com/k8se/quickstart:latest'

var tags = {
  Project: 'HandwritingAI'
  Environment: 'Playground'
  ManagedBy: 'Bicep'
}

module containerRegistry 'modules/container-registry.bicep' = {
  name: 'containerRegistry'
  params: {
    name: acrName
    location: location
    tags: tags
  }
}

module logAnalytics 'modules/log-analytics.bicep' = {
  name: 'logAnalytics'
  params: {
    name: logAnalyticsName
    location: location
    tags: tags
  }
}

module containerAppsEnvironment 'modules/container-apps-env.bicep' = {
  name: 'containerAppsEnvironment'
  params: {
    name: environmentName
    location: location
    logAnalyticsName: logAnalytics.outputs.name
    tags: tags
  }
}

module backend 'modules/container-app.bicep' = if (deployApps) {
  name: 'backend'
  params: {
    name: backendAppName
    location: location
    environmentId: containerAppsEnvironment.outputs.id
    containerName: 'backend'
    image: backendImage
    targetPort: 8000
    externalIngress: false
    cpu: '4.0'
    memory: '8Gi'
    minReplicas: 1
    maxReplicas: 3
    concurrentRequests: '2'
    livePath: '/v1/live'
    readyPath: '/v1/health'
    registryServer: containerRegistry.outputs.loginServer
    envVars: [
      { name: 'DEVICE', value: 'cpu' }
      { name: 'USE_MOCK_ENGINE', value: 'false' }
      { name: 'VOCAB_DIR', value: '/app/data/reference_handwriting/vocabularies' }
    ]
    tags: union(tags, { Tier: 'Backend' })
  }
}

module frontend 'modules/container-app.bicep' = if (deployApps) {
  name: 'frontend'
  params: {
    name: frontendAppName
    location: location
    environmentId: containerAppsEnvironment.outputs.id
    containerName: 'frontend'
    image: frontendImage
    targetPort: 3000
    externalIngress: true
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 2
    concurrentRequests: '50'
    livePath: '/'
    readyPath: '/'
    registryServer: containerRegistry.outputs.loginServer
    envVars: [
      {
        name: 'BACKEND_URL'
        value: 'https://${backendAppName}.internal.${containerAppsEnvironment.outputs.defaultDomain}'
      }
    ]
    tags: union(tags, { Tier: 'Frontend' })
  }
}

// AcrPull is assigned in deploy_apps.sh before the image changes.
// Creating it here races the first revision pull and fails on redeploy
// with RoleAssignmentExists once the CLI assignment already exists.

output acrLoginServer string = containerRegistry.outputs.loginServer
output acrName string = containerRegistry.outputs.name
output logAnalyticsName string = logAnalytics.outputs.name
output environmentDefaultDomain string = containerAppsEnvironment.outputs.defaultDomain
output backendInternalUrl string = 'https://${backendAppName}.internal.${containerAppsEnvironment.outputs.defaultDomain}'
output frontendFqdn string = deployApps ? (frontend.?outputs.fqdn ?? '') : ''
