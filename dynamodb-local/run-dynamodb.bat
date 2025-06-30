@echo off
IF NOT EXIST "data" (
    echo Creating data directory...
    mkdir data
)

echo Starting DynamoDB Local...
java -Djava.library.path=./DynamoDBLocal_lib -jar DynamoDBLocal.jar -sharedDb -dbPath ./data -port 8001
pause