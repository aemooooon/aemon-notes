
### Installing PM2 Globally

To install PM2 globally, use npm:

```bash
sudo npm install -g pm2
```

## Using PM2 to Start Applications

### Starting a Node.js Application

To start a Node.js application with PM2, navigate to your application directory and use the following command:

```bash
pm2 start /path/to/your/app.js --name your-app-name
```

For example:

```bash
pm2 start /home/ubuntu/my-node-app/index.js --name my-node-app
```

### Starting a Python Application

To start a Python application with PM2, use the following command:

```bash
pm2 start /path/to/your/app.py --interpreter python3 --name your-python-app
```

For example:

```bash
pm2 start /home/ubuntu/my-python-app/app.py --interpreter python3 --name my-python-app
```

注意：如果使用了Python虚拟环境，要注意路径
### Setting Up PM2 Startup Script

To set up PM2 to start on system boot:

```bash
pm2 startup
```

This command will generate a command specific to your operating system. Run the generated command to complete the setup.
### Saving the PM2 Process List

To save the list of processes so they restart automatically on reboot:

```bash
pm2 save
```

## Managing Applications with PM2

### Listing Running Processes

To list all running processes managed by PM2, use:

```bash
pm2 list
```

### Stopping a Process

To stop a process, use the process name or ID:

```bash
pm2 stop your-app-name
# or
pm2 stop <process-id>
```

### Restarting a Process

To restart a process, use:

```bash
pm2 restart your-app-name
# or
pm2 restart <process-id>
```

### Deleting a Process

To delete a process, use:

```bash
pm2 delete your-app-name
# or
pm2 delete <process-id>
```

### Viewing Logs

To view logs for all processes:

```bash
pm2 logs
```

To view logs for a specific process:

```bash
pm2 logs your-app-name
# or
pm2 logs <process-id>
```

## PM2 Common Commands

### Reloading PM2 Configuration

To reload PM2 configuration without downtime (useful for zero-downtime restarts):

```bash
pm2 reload all
```

### Viewing Process Details

To view detailed information about a specific process:

```bash
pm2 describe your-app-name
# or
pm2 describe <process-id>
```

### Monitoring Resource Usage

To monitor resource usage (CPU, Memory) of your processes:

```bash
pm2 monit
```

# Add airflow to EC2 service via PM2
`start_airflow.sh`

```bash
#!/bin/bash

# 设置 AIRFLOW_HOME
export AIRFLOW_HOME=/home/ubuntu/Data-collection-service

# 激活虚拟环境
source /home/ubuntu/Data-collection-service/airflow_env/bin/activate

# 停止所有正在运行的 Airflow 服务
pkill -f "airflow webserver"
pkill -f "airflow scheduler"
pkill -f "airflow worker"

# 确保所有进程已停止
sleep 5

# 再次确认是否有残留进程
PIDS=$(ps -ef | grep "airflow" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "Killing remaining Airflow processes..."
  kill -9 $PIDS
fi

# 启动 Airflow webserver
airflow webserver --port 8080 &

# 启动 Airflow scheduler
airflow scheduler &

```

```
chmod +x start_airflow.sh
pm2 start start_airflow.sh --name airflow
pm2 startup
pm2 save
pm2 status
```