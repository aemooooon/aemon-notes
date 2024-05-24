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