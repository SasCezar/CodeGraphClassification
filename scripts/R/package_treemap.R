library(treemap)


project = 'apache|zookeeper'
#project = 'Waikato|weka-3.8'
top = '5'
asg = '1'

dt <- read.csv(sprintf("/home/sasce/PycharmProjects/CodeGraphClassification/scripts/python/%s_top_%s_assign_%s.csv", project, top, asg))

treemap(dt,
        index=c("label","name"),
        vSize="weight",
        type="index",
        palette = "Set1",                        # Select your color palette from the RColorBrewer presets or make your own.
        title=paste(project, top, asg, sep=" "),                      # Customize your title
        fontsize.labels=c(15,12),                # size of labels. Give the size per level of aggregation: size for group, size for subgroup, sub-subgroups...
        fontcolor.labels=c("white","black"),    # Color of labels
        fontface.labels=c(2,1),                  # Font of labels: 1,2,3,4 for normal, bold, italic, bold-italic...
        align.labels=list(
          c("left", "top"),
          c("center", "center")
        ),                                   # Where to place labels in the rectangle?
        overlap.labels=0.5,                      # number between 0 and 1 that determines the tolerance of the overlap between labels. 0 means that labels of lower levels are not printed if higher level labels overlap, 1  means that labels are always printed. In-between values, for instance the default value .5, means that lower level labels are printed if other labels do not overlap with more than .5  times their area size.
        inflate.labels=F,                        # If true, labels are bigger when rectangle is bigger.
)
