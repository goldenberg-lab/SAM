# Script to show threshold trade-off
library(ggplot2)
library(cowplot)
library(dplyr)
library(tibble)
library(tidyr)
# Baseline dist for class dist (unit variance)
mu1 = +1
mu0 = -1

t_seq = seq(-4,4,0.01)
sens = sapply(t_seq, function(tt) pnorm(tt,mean=mu1,lower.tail = F))
spec = sapply(t_seq, function(tt) pnorm(tt,mean=mu0,lower.tail = T))
df = tibble(thresh=t_seq, sens=sens, fpr=1-spec,tt='gt')

nsim = 100
n1 = 100
n0 = 100
holder = list()
set.seed(nsim)
for (i in seq(nsim)) {
  if (i %% 100 == 0) {
    print(i)
  }
  s1 = rnorm(n1, mu1)
  s0 = rnorm(n0, mu0)
  esens = sapply(t_seq, function(tt) mean(s1 > tt))
  espec = sapply(t_seq, function(tt) mean(s0 < tt))
  edf = tibble(thresh=t_seq, sens=esens, fpr=1-espec, sim=i)
  holder[[i]] = edf
}
res_sim = do.call('rbind',holder) %>% mutate(tt='sim')

res_all = rbind(mutate(df,tt='gt',sim=0),mutate(res_sim,tt='sim'))
lblz = c('Ground truth','Simulation')
# Find the specificity target
target_fpr = 0.1
df_point = res_sim %>% 
  filter(fpr < target_fpr) %>% 
  group_by(sim) %>% 
  filter(thresh == min(thresh))
thresh_dist = df_point$thresh

# Remove duplicated threshold results
res_sim2 = res_sim %>% 
  group_by(tt,sim,sens,fpr) %>% 
  summarise(thresh=min(thresh))

# df_point = data.frame(fpr=target_fpr,sens=thresh_dist)
df_txt = data.frame(fpr=0.2,sens=0.5,txt='10% FPR')
gg_auroc = ggplot() + 
  theme_bw() + labs(x='FPR',y='TPR') + 
  geom_text(aes(x=fpr,y=sens,label=txt),data=df_txt,color='blue',size=5) + 
  geom_point(aes(x=fpr,y=sens),color='grey',data=df_point) + 
  geom_line(data=res_sim2,aes(x=fpr,y=sens,color=tt,group=sim),alpha=0.5) + 
  geom_line(data=df,aes(x=fpr,y=sens,color=tt),size=2) + 
  scale_color_manual(name='Type',values=c('black','grey'),labels=lblz) + 
  ggtitle('Ground truth ROC curve with empirical distribution') + 
  theme(legend.position = c(0.8,0.3)) + 
  geom_segment(aes(y=0,yend=1,x=0,xend=1),color='black',linetype=2,size=1) +
  geom_vline(xintercept = target_fpr, color='blue')

# Compare dist to ground truth
thresh_gt = qnorm(p=1-target_fpr,mean=mu0)

df_txt_thresh = data.frame(x=thresh_gt*1.3,y=9,txt='10% FPR')
gg_thresh = ggplot(df_point,aes(x=thresh)) + theme_bw() + 
  geom_text(aes(x=x,y=y,label=txt),data=df_txt_thresh,color='blue',size=5) + 
  geom_histogram(color='darkgrey',fill='grey',bins=25) + 
  geom_vline(xintercept=thresh_gt, color='blue') + 
  ggtitle('Ground truth threshold with empirical distribution') + 
  labs(y='Frequency',x='Empirical threshold')

gg_both = plot_grid(gg_auroc,gg_thresh,nrow=1,labels=c('A','B'))
ggsave2(file.path('figures','roc_thresh.png'),gg_both, width=10,height=4)

