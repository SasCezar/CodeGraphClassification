library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(RColorBrewer)
library(geomtextpath)
library(ggpubr)
library(paletteer)

size = 11

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/human_eval_agreement.csv")
dt <- dt %>% 
  filter(metric == 'kappa') %>%
  mutate(level = fct_relevel(level, 'project', 'package', 'file')) %>%
  mutate(across(1, str_to_title))%>%
  mutate(across(3, round, 2))

p <- ggplot(dt, aes(y = score , x = level,  fill= level, label=score)) +
  geom_bar(stat="identity", show.legend = FALSE) +
  geom_text(size = size/3, position = position_stack(vjust = 1.02)) +
  scale_fill_brewer(palette="Set2") +
  theme(text = element_text(size = size),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  xlab(element_blank()) + ylab("Kappa") 
p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/human_eval_agreement.pdf', width=4, height=4)

# dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/human_eval_results.csv")
# dt <- dt %>% 
#   mutate(level = fct_relevel(level, 'project', 'package', 'file'))
# 
# p <- ggplot(dt, aes(y = percent , x = label,  fill= level)) +
#   geom_bar(stat="identity") +
#   facet_nested(~ level) +
#   theme(text = element_text(size = size),
#         axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
#   xlab("Label") + ylab("Percent") 
# p

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/human_eval_results.csv")
dt$label <- as.factor(dt$label)
dt <- dt %>% 
  mutate(level = fct_relevel(level, 'project', 'package', 'file')) %>%
  mutate(across(1, str_to_title)) %>%
  mutate(group = fct_relevel(group, 'Incorrect', 'Correct'))%>% 
  mutate(label = fct_relevel(label, '3', '2', '1', '0')) %>%
  mutate(across(5, round, 2))

dt$percent <- 100 * dt$percent

dt_proj <- dt %>% 
  filter(level == 'Project') %>%
  mutate(label = fct_relevel(label, '0', '1')) # %>%
  #mutate(group = replace(group, group == 'Correct', 'Relevant')) %>%
  #mutate(group = replace(group, group == 'Incorrect', 'Yake'))

dt <- dt %>% 
  filter(level != 'Project') %>%
  mutate(level = fct_relevel(level, 'Package', 'File'))


size_pack = 11
p <- ggplot(dt, aes(y = percent , x = group,  fill=label, label=percent)) +
  geom_bar(position="stack", stat="identity") +
  #scale_fill_brewer(palette="Paired") +
  scale_fill_manual(values = c("#E4F0DA", "#B2DF8A", "#33A02C", "#E31A1C")) + 
  geom_text(size = size/3, position = position_stack(vjust = 0.5))+
  facet_nested(~ level) +
  theme(text = element_text(size = size_pack),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  xlab(element_blank()) + ylab("Percent") + labs(fill = "Position")
p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/human_eval_results.pdf', width=6, height=4)

ppp <- ggplot(dt_proj, aes(y = percent , x = group,  fill=label)) +
  geom_bar(position="stack", stat="identity", show.legend = FALSE) +
  #scale_fill_brewer(palette="Paired") +
  scale_fill_manual(values = c("#E31A1C", "#33A02C")) + 
  theme(text = element_text(size = 17),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  ylab("Percent") + xlab(element_blank()) 
ppp

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/human_eval_project_percent.pdf', width=3, height=4)

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/LF_project_new_topics.csv")

dt$num <- as.factor(dt$num)
pnt <- ggplot(dt, aes(y = count , x=num, fill=num)) +
  geom_bar(stat="identity", show.legend = FALSE) +
  scale_fill_paletteer_d("ggsci::springfield_simpsons") +
  geom_textvline(label = "3.24", xintercept = 4.24, # Is 4.24 to compensate for the 1 extra space occupied by the 0 (the entries are discrete not continuous)
             color = "black", size = 4, hjust = .96) +
  #geom_boxplot(width = width, fill="white", outlier.shape = NA, show.legend = FALSE) +
  theme(text = element_text(size = 17),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylab("Projects") +  xlab(element_blank())# xlab("Number of new labels") 

pnt

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/human_eval_project_new.pdf', width=8, height=4)


pg1 <- ggplotGrob(ppp)
pg2 <- ggplotGrob(pnt)

maxHeight = grid::unit.pmax(pg1$heights, pg2$heights)
pg1$heights <- as.list(maxHeight)
pg2$heights <- as.list(maxHeight)
ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/human_eval_project_joined_a.pdf', plot=pg1, width=3, height=4)
ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/human_eval_project_joined_b.pdf', plot=pg2, width=7, height=4)

ggarrange(pg1, pg2, ncol = 2, widths = c(1,5))

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/human_eval_project_joined.pdf', width=9, height=4)