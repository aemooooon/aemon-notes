---
title: 
draft: false
tags:
  - git
date: 2024-06-06
---


```bash
# view logs/records and 临时暂存工作搞点别的再回来
gloga
gwip
gunwip

# example
git add .
git gwip
git checkout other-branch
git checkout your-branch
git gunwip
git add .
git commit -m "Your actual commit message"

# 在前一个commit的基础上修改或者保留message，但是hash会变
git commit --amend
```

```bash
git remote prune origin # 刷新本地仓库保持与远程仓库的改动的同步

git push origin --delete [branch_name] # 删除远程分支 

#When you are making a pull request, you will see that there is nothing to compare against. Thanks to Jamie, here is a command that will fix that:

git merge main --allow-unrelated-histories

# Make sure you run this from the practical branch.
```