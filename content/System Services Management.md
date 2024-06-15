---
title: 
draft: false
tags:
  - service
date: 2024-06-01
---



`sudo vim /etc/systemd/system/individual-example.service`

```bash
[Unit]
Description=DATA472 Individual Example App
After=network.target

[Service]
ExecStart=/usr/bin/node /home/ubuntu/hua/Individual-example/app.js
Restart=always
User=ubuntu
Environment=PATH=/usr/bin:/usr/local/bin
Environment=NODE_ENV=production
WorkingDirectory=/home/ubuntu/hua/Individual-example/

[Install]
WantedBy=multi-user.target
```

```
sudo systemctl daemon-reload
sudo systemctl start individual-example
sudo systemctl status individual-example
sudo systemctl enable individual-example
journalctl -u individual-example # check logs
```
