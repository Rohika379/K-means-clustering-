


library(readxl)
crime <- read.csv("C:\\Users\\sanjay\\OneDrive\\Desktop\\360digiTMG assignment\\Datasets_Kmeans\\crime_data (1).csv")
mydata <- crime [,-1 ]

# Normalize the data
normalized_data <- scale(mydata[,])

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
final <- data.frame(fit$cluster, mydata) # Append cluster membership

aggregate(mydata[, ], by = list(fit$cluster), FUN = mean)
