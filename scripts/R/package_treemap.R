library(treemap)
library(dplyr)
library(stringr)
library(microViz)
library(stringr)

#project = 'apache|zookeeper'
projects = list('Waikato|weka-3.8', 'mickleness|pumpernickel')
#project = 'mickleness|pumpernickel'
tops = list('5', '10', '20')
asg = '1'

for (project in projects) {
  x <- distinct_palette(n = 20, pal = "brewerPlus")
  t <- sample(x, 21)
  for (top in tops) {
    dt <- read.csv(sprintf("/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/labelled_graph/ensemble/best/voting/none/none/%s_top_%s_assign_%s.csv", project, top, asg))
    dt <- dt %>% 
      mutate(label = replace(label, label == 'Other', 'Other Labels')) %>%
      mutate(across('name', \(x) str_replace(x, "weka.", "")))%>% 
      mutate(across('name', \(x) str_replace(x, "com.pump.", "")))
    
    pdf(sprintf("/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/annotations/%s_top_%s_assign_%s_labelled.pdf", str_replace(project, '[|]', '-'), top, asg), width=8.5, height=7)
    p1 <- treemap(dt,
                  index=c("label", "name"),
                  vSize="weight",
                  type="index",
                  palette = t,                        # Select your color palette from the RColorBrewer presets or make your own.
                  title='',                      # Customize your title
                  fontsize.labels=c(22,18),                # size of labels. Give the size per level of aggregation: size for group, size for subgroup, sub-subgroups...
                  fontcolor.labels=c("white","black"),    # Color of labels
                  fontface.labels=c(2,1),                  # Font of labels: 1,2,3,4 for normal, bold, italic, bold-italic...
                  align.labels=list(
                    c("left", "top"),
                    c("center", "center")
                  ),                                   # Where to place labels in the rectangle?
                  overlap.labels=0,                      # number between 0 and 1 that determines the tolerance of the overlap between labels. 0 means that labels of lower levels are not printed if higher level labels overlap, 1  means that labels are always printed. In-between values, for instance the default value .5, means that lower level labels are printed if other labels do not overlap with more than .5  times their area size.
                  inflate.labels=F,                        # If true, labels are bigger when rectangle is bigger.
                  border.col=c("black","white"),             # Color of borders of groups, of subgroups, of subsubgroups ....
                  border.lwds=c(5,2)                         # Width of colors
    )
    
    
    dev.off()
  }
  
  
  pdf(sprintf("/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/annotations/%s_blank.pdf", str_replace(project, '[|]', '-')), width=8.5, height=7)
  
  dt <- dt %>% 
    mutate(label = '') 
  
  uncolored <- colorRampPalette(colors = c('lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey'))(1)
  treemap(dt,
          index=c("label","name"),
          vSize="weight",
          type="index",
          palette =  uncolored,                        # Select your color palette from the RColorBrewer presets or make your own.
          title='',                      # Customize your title
          fontsize.labels=c(0,18),                # size of labels. Give the size per level of aggregation: size for group, size for subgroup, sub-subgroups...
          fontcolor.labels=c("white","black"),    # Color of labels
          fontface.labels=c(2,1),                  # Font of labels: 1,2,3,4 for normal, bold, italic, bold-italic...
          align.labels=list(
            c("left", "top"),
            c("center", "center")
          ),                                   # Where to place labels in the rectangle?
          overlap.labels=0.1,                      # number between 0 and 1 that determines the tolerance of the overlap between labels. 0 means that labels of lower levels are not printed if higher level labels overlap, 1  means that labels are always printed. In-between values, for instance the default value .5, means that lower level labels are printed if other labels do not overlap with more than .5  times their area size.
          inflate.labels=F,                        # If true, labels are bigger when rectangle is bigger.
          border.col=c("black","white"),             # Color of borders of groups, of subgroups, of subsubgroups ....
          border.lwds=c(5,2)                         # Width of colors
  )
  
  dev.off()
}