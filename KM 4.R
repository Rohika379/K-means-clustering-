
#File input

library(readxl)

telco <- read_excel(file.choose())

#Use of daisy function

library(caret)
dmy <- dummyVars("~ .",data =telco[c(22,24)],fullRank = TRUE )
dat_transformed <- data.frame(predict(dmy,newdata =telco [c(22,24)]))
finaldata <- cbind(telco[c(22,24)],dat_transformed)

# Normalize the data
normalized_data <- scale(telco[,c(9,13,25,26,30)])

# Elbow curve to decide the k value
twss <- NULL
for (i in 2:8) {
     twss <- c(twss, kmeans(normalized_data, centers = i)$tot.withinss)
     }
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")

title(sub = "K-Means Clustering Scree-Plot")


# 3 Cluster Solution
fit <- kmeans(normalized_data, 3) 
str(fit)
fit$cluster
final <- data.frame(fit$cluster, telco) # Append cluster membership

aggregate(telco[, ], by = list(fit$cluster), FUN = mean)
