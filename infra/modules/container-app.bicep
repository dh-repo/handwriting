param name string
param location string
param environmentId string
param containerName string
param image string
param targetPort int
param externalIngress bool
param cpu string
param memory string
param minReplicas int
param maxReplicas int
param concurrentRequests string
param envVars array = []
param livePath string
param readyPath string
param registryServer string
param tags object = {}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    environmentId: environmentId
    configuration: {
      activeRevisionsMode: 'Single'
      maxInactiveRevisions: 5
      ingress: {
        external: externalIngress
        targetPort: targetPort
        transport: 'auto'
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
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
          probes: [
            {
              type: 'Startup'
              httpGet: {
                path: livePath
                port: targetPort
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              failureThreshold: 36
            }
            {
              type: 'Liveness'
              httpGet: {
                path: livePath
                port: targetPort
              }
              initialDelaySeconds: 15
              periodSeconds: 30
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: readyPath
                port: targetPort
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: concurrentRequests
              }
            }
          }
        ]
      }
    }
  }
  tags: tags
}

output name string = containerApp.name
output fqdn string = containerApp.properties.configuration.ingress.fqdn
output systemAssignedMIPrincipalId string = containerApp.identity.principalId
