library(afex)
library(emmeans)

.Random.seed <- 1

data <- read.csv('/home/simon/Data/mri-augmentation/spreadsheets/Dice/dice_table_big_alt_mod.csv')
data$bone <- factor(data$bone)
data$severity <- factor(data$severity)
data$model <- factor(data$model)

model <- mixed(dice ~ bone + severity + model + (1 | subject), data = data, check_contrasts = FALSE) # check_contrasts = FALSE to enfore reference level coding as opposed to sum-to-zero coding (see https://github.com/singmann/afex/issues/4)

print('Coefficients')
print(summary(model))
# How to interpret these results:
# Each factor has a reference level (proximal_femur, reference, baseline) and the coefficients are the differences between the reference level and the other levels.
# I.e. if the coefficient for the mr model is 0.1, the mr model performs 0.1 better than the baseline model.
# - multiple comparisons
# general workflow for multiple comparisons:
# - fit model mixed(dice ~ bone * severity * model + (1 | subject))
# - create emmeans emmeans(model, ~model), emmeans(model, ~model|bone), emmeans(model, ~model|bone*severity) etc.
# - create contrast contrast(emmeans, "tukey")
model <- mixed(dice ~ bone * severity * model + (1 | subject), data = data) # here I want the interaction between factors

print('##################')
print('Multiple comparisons')
print('##################')

# model
print('##################')
print('Model - emmeans')
print(emmeans(model, ~model)) # difference between models averaged over bone and severity

print('Model - contrast')
print(contrast(emmeans(model, ~model), "tukey")) # difference between models averaged over bone and severity

# model|severity
print('##################')
print('Model|Severity - emmeans')
print(emmeans(model, ~model|severity)) # difference between models for each severity level, averaged over bone

print('Model|Severity - contrast')
print(contrast(emmeans(model, ~model|severity), "tukey")) # difference between models for each severity level, averaged over bone


# model|bone
print('##################')
print('Model|Bone - emmeans')
print(emmeans(model, ~model|bone)) # difference between models for each bone level, averaged over severity

print('Model|Bone - contrast')
print(contrast(emmeans(model, ~model|bone), "tukey")) # difference between models for each bone level, averaged over severity

# model|bone*severity
print('##################')
print('Model|Bone*Severity - emmeans')
print(emmeans(model, ~model|bone*severity)) # difference between models for each bone and severity level

print('Model|Bone*Severity - contrast')
print(contrast(emmeans(model, ~model|bone*severity), "tukey")) # difference between models for each bone and severity level
