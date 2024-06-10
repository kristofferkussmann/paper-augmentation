library(afex)
library(emmeans)

.Random.seed <- 1

data <- read.csv('/home/simon/Data/mri-augmentation/spreadsheets/Torsion/torsion_table_long_format_v2_irm.csv')
data$side <- factor(data$side)
data$severity <- factor(data$severity)
data$type <- factor(data$type)
data$reader <- factor(data$reader)

model <- mixed(torsion ~ type:(side + severity + reader) - 1 + (type - 1 | subject), data = data, check_contrasts = FALSE)

print(summary(model))

#data <- read.csv('/home/simon/Data/mri-augmentation/spreadsheets/Torsion/torsion_table_long_format_v2.csv')
#ata$side <- factor(data$side)
#data$severity <- factor(data$severity)
#data$type <- factor(data$type)
#data$reader <- factor(data$reader)

model <- mixed(torsion ~ type:(side * severity * reader) - 1 + (type - 1 | subject), data = data)

print(emmeans(model, ~reader | side * severity))
print(contrast(emmeans(model, ~reader | side * severity), "tukey"))