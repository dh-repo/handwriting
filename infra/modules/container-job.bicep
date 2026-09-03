param name string
param location string
param environmentId string
param containerName string = 'training-worker'
param image string
param cpu string = '2.0'
param memory string = '4Gi'
param registryServer string
param envVars array = []
param triggerType string = 'Manual'
param cronExpression string = ''
param replicaTimeout int = 3600
param replicaRetryLimit int = 1
param tags object = {}

resource containerAppJob 'Microsoft.App/jobs@2024-03-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    environmentId: environmentId
    configuration: {
      triggerType: triggerType
      replicaTimeout: replicaTimeout
      replicaRetryLimit: replicaRetryLimit
      scheduleTriggerConfig: triggerType == 'Schedule' ? {
        cronExpression: cronExpression
        parallelism: 1
      } : null
      manualTriggerConfig: triggerType == 'Manual' ? {
        parallelism: 1
        replicaCompletionCount: 1
      } : null
      registries: [
        {
          server: registryServer
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: containerName
          image: image
          env: envVars
          resources: {
            cpu: json(cpu)
            memory: memory
          }
        }
      ]
    }
  }
}

output id string = containerAppJob.id
output name string = containerAppJob.name
