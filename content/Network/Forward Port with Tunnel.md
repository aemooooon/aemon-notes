---
title: 
draft: false
tags:
---

## 命令行
```bash
brew install autossh

autossh -M 0 -i /Users/hua/.ssh/DATA472-jhs348-grpkey.pem -L 5434:data472-jre141-groupcollection-1.cyi9p9kw8doa.ap-southeast-2.rds.amazonaws.com:5432 ubuntu@3.25.53.109
```

## 脚本

 connect-RDS.sh
```bash
#!/bin/bash
autossh -M 0 -i /Users/hua/.ssh/DATA472-jhs348-grpkey.pem -L 5434:data472-jre141-groupcollection-1.cyi9p9kw8doa.ap-southeast-2.rds.amazonaws.com:5432 ubuntu@3.25.53.109

chmod +x connect-RDS.sh

./connect-RDS.sh
```


## 配置文件
```bash
Host myrds
	HostName ec2-instance-public-dns 
	User ec2-user 
	IdentityFile path/to/your-key-file.pem 
	LocalForward 5433 rds-endpoint:5432
```

`ssh myrds`
