## Generate Keys
```bash
sudo apt update  
sudo apt install openssh-client
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
cp id_rsa.pub to GitHub setup
```
## Extend Session Time
Add it to `~/.ssh/config`
```bash
Host *
  ServerAliveInterval 60
  ServerAliveCountMax 120
```

