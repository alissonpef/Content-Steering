docker build -t eduardogama/steering:latest .
docker run -p 30500:30500 --rm --name steering -t eduardogama/steering:latest
