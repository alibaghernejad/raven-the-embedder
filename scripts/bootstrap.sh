#!/bin/bash

# If you're using this project alongside **Retraven**, there's no need to create these resources manually.
# — everything is provisioned automatically.  
# Check out the main repository here: [Retraven](https://github.com/alibaghernejad/retraven)

docker run -d -p 6333:6333 qdrant/qdrant:v1.13.6