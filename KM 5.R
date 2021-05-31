
#File input

auto <- read.csv("C:\\Users\\sanjay\\OneDrive\\Desktop\\360digiTMG assignment\\Dataset_Assignment Clustering\\AutoInsurance.csv")


#categorical encoder
library(caret)
dmy <- dummyVars("~ .",data = auto[c(8,9,11,12)],fullRank = TRUE)
dat_tra <- data.frame(predict(dmy,newdata = auto[c(8,9,11,12)]))
final <- cbind(auto[-c(8,9,11,12)],dat_tra)

#columns for clustering

cls <- final[,c(3,8,9,10,11,13,18)]
summary(cls)

#normalisation

normalized_data <- scale(cls)
summary(normalized_data)

 
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
final <- data.frame(fit$cluster, auto) # Append cluster membership

aggregate(auto[,], by = list(fit$cluster), FUN = mean)
