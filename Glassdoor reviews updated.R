
library(readr)
Uncleaned_DS_jobs <- read_csv("Downloads/Uncleaned_DS_jobs.csv")
View(Uncleaned_DS_jobs)

uncleaned <- subset(Uncleaned_DS_jobs, select = -c(index))
head(Uncleaned_DS_jobs)

# shape rows * colummns
dim(Uncleaned_DS_jobs)

# data types
str(Uncleaned_DS_jobs)

print(dim(uncleaned[duplicated(Uncleaned_DS_jobs), ]))
Uncleaned_DS_jobs[duplicated(Uncleaned_DS_jobs), ]

# find distinct values for the rows and columns
Uncleaned_DS_jobs <- distinct(Uncleaned_DS_jobs)

# Removing numbers from company name

Uncleaned_DS_jobs$`Company Name` <- sub("\n.*", "", Uncleaned_DS_jobs$`Company Name`)

# finding salary ranges with their no of data scientist roles
table(Uncleaned_DS_jobs$`Salary Estimate`)

Uncleaned_DS_jobs$min_salary <- 0
Uncleaned_DS_jobs$max_salary <- 0
Uncleaned_DS_jobs$avg_salary <- 0

for (i in 1:nrow(Uncleaned_DS_jobs)) {
  sal_str <- as.character(Uncleaned_DS_jobs[i, "Salary Estimate"])
  sal_parts <- strsplit(sal_str, " ")[[1]]
  if (length(sal_parts) > 1) {
    min_sal_str <- strsplit(sal_parts[1], "-")[[1]][1]
    max_sal_str <- strsplit(sal_parts[1], "-")[[1]][2]
  } else {
    sal_parts_e <- strsplit(sal_parts[1], "(E")[1]
    min_sal_str <- strsplit(sal_parts_e, "-")[[1]][1]
    max_sal_str <- strsplit(sal_parts_e, "-")[[1]][2]
  }
  min_salary <- as.integer(gsub("\\$|K", "", min_sal_str))
  max_salary <- as.integer(gsub("\\$|K", "", max_sal_str))
  avg_salary <- mean(c(min_salary, max_salary))
  
  Uncleaned_DS_jobs[i, "min_salary"] <- min_salary
  Uncleaned_DS_jobs[i, "max_salary"] <- max_salary
  Uncleaned_DS_jobs[i, "avg_salary"] <- avg_salary
  Uncleaned_DS_jobs[i, "Salary Estimate"] <- paste(min_salary, "-", max_salary)
}
 
str(Uncleaned_DS_jobs)

# Extracting job_state from Location Column and replacing Full Names with appropriate state code and replacing remote with NA

table(sapply(strsplit(Uncleaned_DS_jobs$Location, ","), tail, n = 1))

# Extract state from Location column
Uncleaned_DS_jobs$job_state <- sapply(strsplit(Uncleaned_DS_jobs$Location, ","), function(x) trimws(x[length(x)]))

# Replace state names with abbreviations and drop rows with NaN
Uncleaned_DS_jobs$job_state[Uncleaned_DS_jobs$job_state %in% c("United States", "Texas", "California", "New Jersey", "Remote", "Utah")] <- c("US", "TX", "CA", "NJ", NA, "UT")
Uncleaned_DS_jobs <- na.omit(Uncleaned_DS_jobs)

# Reset the row indices
rownames(Uncleaned_DS_jobs) <- NULL

Uncleaned_DS_jobs 

table(Uncleaned_DS_jobs$job_state)

Uncleaned_DS_jobs$Rating <- ifelse(Uncleaned_DS_jobs$Rating == -1.0, 0, Uncleaned_DS_jobs$Rating)
Uncleaned_DS_jobs <- Uncleaned_DS_jobs[order(Uncleaned_DS_jobs$Rating, decreasing=FALSE),]
head(Uncleaned_DS_jobs)


# checking if the headquarters and job state is same for every company(if yes(1) then same state if not 0)

Uncleaned_DS_jobs$Headquarters1 <- sapply(strsplit(as.character(Uncleaned_DS_jobs$Headquarters), ","), function(x) trimws(x[length(x)]))
Uncleaned_DS_jobs$same_state <- ifelse(Uncleaned_DS_jobs$Headquarters1 == Uncleaned_DS_jobs$job_state, 1, 0)
Uncleaned_DS_jobs$Headquarters1 <- NULL


# calculating company age from Founded year

Uncleaned_DS_jobs$Founded <- ifelse(Uncleaned_DS_jobs$Founded == -1, 0, Uncleaned_DS_jobs$Founded)
Uncleaned_DS_jobs$company_age <- 2023 - Uncleaned_DS_jobs$Founded


strsplit(Uncleaned_DS_jobs$`Job Description`[1], "\n\n", fixed = TRUE)[[1]]

head(sort(table(Uncleaned_DS_jobs$`Job Title`), decreasing = TRUE), 100)

seniority <- function(job_title) {
  job_title <- tolower(job_title)
  snr <- c('sr', 'senior', 'lead', 'principal', 'vp', 'vice president', 'director')
  for (i in snr) {
    if (grepl(i, job_title)) {
      return("senior")
    }
  }
  if (grepl("jr", job_title)) {
    return("junior")
  }
  return("na")
}


Uncleaned_DS_jobs$Seniority <- sapply(Uncleaned_DS_jobs$`Job Title`, seniority)


# to check if the function is working 
filtered_data <- subset(Uncleaned_DS_jobs, `Job Title` == "Vice President, Biometrics and Clinical Data Management")

categorize_job_title <- function(job_title) {
  job_title <- tolower(job_title)
  if (grepl("data scientist", job_title)) {
    return("Data Scientist")
  } else if (grepl("data engineer", job_title)) {
    return("Data Engineer")
  } else if (grepl("analyst", job_title)) {
    return("Analyst")
  } else if (grepl("machine learning engineer", job_title)) {
    return("Machine Learning Engineer")
  } else if (grepl("manager", job_title)) {
    return("Manager")
  } else if (grepl("director", job_title)) {
    return("Director")
  } else if (grepl("vice president", job_title)) {
    return("Vice President")
  } else {
    return("Other")
  }
}

# Apply the function to categorize job titles
Uncleaned_DS_jobs$categorize_job_title <- sapply(Uncleaned_DS_jobs$`Job Title`, categorize_job_title)

# View the first few rows of the updated data
head(Uncleaned_DS_jobs)


str(Uncleaned_DS_jobs)

write.csv(Uncleaned_DS_jobs, "Cleaned_DS_Jobs_2.csv", row.names=FALSE)

## Loading the cleaned dataset

cleaned_DS_jobs <- read_csv("Cleaned_DS_Jobs_2.csv")

summary(cleaned_DS_jobs)

cleaned_DS_jobs <- cleaned_DS_jobs[, !(names(cleaned_DS_jobs) %in% c("Competitors"))]

# Drop rows with -1 in Size column
# cleaned_DS_jobs <- cleaned_DS_jobs[cleaned_DS_jobs$Size != "-1",]
  
# Drop rows with -1 in Founded column
 cleaned_DS_jobs <- cleaned_DS_jobs[cleaned_DS_jobs$Founded != -1,]

# Drop rows with -1 in Industry column
# cleaned_DS_jobs <- cleaned_DS_jobs[cleaned_DS_jobs$Industry != "-1",]

# Drop rows with -1 in Sector column
 cleaned_DS_jobs <- cleaned_DS_jobs[cleaned_DS_jobs$Sector != -1,]



cleaned_DS_jobs$salary_diff <- cleaned_DS_jobs$max_salary - cleaned_DS_jobs$min_salary

library(ggplot2)

# Create a bar plot for the salary range counts
ggplot(data = cleaned_DS_jobs, aes(x = factor(`Salary Estimate`))) +
  geom_bar(fill = "Orange") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 13),
        axis.text.y = element_text(size = 13),
        plot.title = element_text(size = 16, face = "bold"),
        axis.title.y = element_text(size = 13),
        legend.position = "none") +
  labs(title = "Salary Range Counts", y = "Count")



ggplot(data = cleaned_DS_jobs, aes(x=categorize_job_title)) +
  geom_bar(stat = "Count",fill="Red") +
  labs(title="Job Title Count", y="Count", x="Job Title") +
  theme(axis.text.x = element_text(angle=45, vjust=1, hjust=1, size=13)) +
  theme(plot.title = element_text(size=16, face="bold"))


ggplot(data = cleaned_DS_jobs, aes(x = categorize_job_title, y = avg_salary/n_distinct(categorize_job_title))) +
  geom_bar(stat="identity", fill="midnightblue") +
  labs(title="Avg Salary for each Job", x="Job Simplified", y="Avg Salary") +
  theme(plot.title=element_text(size=16, face="bold"),
        axis.text.x=element_text(angle=45, vjust=0.5, hjust=1, size=13),
        axis.text.y=element_text(size=13),
        axis.title=element_text(size=13))


ggplot(cleaned_DS_jobs, aes(x = categorize_job_title, y = max_salary)) +
  geom_bar(stat = "identity", fill = "midnightblue") +
  labs(title = "Max Salary for each Job", x = "Job Simplified", y = "Max Salary") +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 1, size = 13),
        axis.text.y = element_text(size = 13),
        axis.title = element_text(size = 13))

library(dplyr)



cleaned_DS_jobs <- read_csv("Cleaned_DS_Jobs_2.csv")

summary(cleaned_DS_jobs)

cleaned_DS_jobs <- cleaned_DS_jobs[, !(names(cleaned_DS_jobs) %in% c("Competitors"))]

cleaned_DS_jobs %>%
  count(`Company Name`) %>%
  top_n(5, n) %>%
  ggplot(aes(x = n, y = `Company Name`)) +
  geom_col(fill = "midnightblue") +
  labs(title = "Top 5 Companies by Frequency", x = "Frequency", y = "Company Name") +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 13),
        axis.text.y = element_text(size = 13),
        axis.title = element_text(size = 13),
        panel.background = element_rect(fill = "white"),
        panel.grid.major = element_line(color = "gray80"),
        panel.grid.minor = element_blank()) +
  coord_flip()


library(corrplot)

numeric_df <- select_if(cleaned_DS_jobs, is.numeric)
# Create correlation matrix
cor_mat <- cor(numeric_df)

# Set up plot
par(plt = c(0.1, 0.9, 0.1, 0.9))

# correlation between variables in percent
corrplot(cor_mat, method = "color", type = "lower", tl.col = "black", tl.srt = 45, tl.cex = 0.8,
         addCoef.col = "black", addCoefasPercent = TRUE, 
         col = colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))(200))


library(readr)


# Function to extract the upper limit value from a range
extract_upper_limit <- function(revenue) {
  # Convert "Unknown / Non-Applicable" to NA
  if (revenue == "Unknown / Non-Applicable") {
    return(NA)
  }
  else {
  
  revenue_modified <- gsub("\\$(\\d+) to \\$(\\d+) (million|billion) \\(USD\\)", "\\1-\\2", revenue)
  
  print(revenue_modified)

  # Example vector with values
  values <- c("Unknown / Non-Applicable", "1-2", "100-500", "$10+ billion (USD)", "2-5", "$500 million to $1 billion (USD)", "5-10", "10-25", "25-50", "50-100", "1-5", "5-10", "Less than $1 million (USD)")
  
  # Function to extract upper limit of the range
  extract_upper_limit <- function(value) {
    if (grepl("-", value)) {
      upper_limit <- sub(".*-(.*)", "\\1", value)
      return(upper_limit)
    } else {
      return(value)
    }
  }
  
  # Apply the extraction function to the values
  converted_values <- sapply(values, extract_upper_limit)
  
  converted_values <- as.integer(converted_values)
  # Print the converted values
  print(converted_values)
  
  # Convert million to billion
  if (grepl("million", revenue_modified)) {
    upper_limit <- as.numeric(upper_limit) / 1000
  }
  
  return(upper_limit)
}
}
# Apply the function to revenue values
revenue_upper_limit <- sapply(revenue_modified, extract_upper_limit)

print(revenue_upper_limit)

## Performing some of the descriptive statistics to the data

num_vars <- c("Rating", "Revenue", "min_salary", "max_salary", "avg_salary")
num_stats <- data.frame(
  Variable = num_vars,
  Mean = sapply(cleaned_DS_jobs[num_vars], mean, na.rm = TRUE),
  Median = sapply(cleaned_DS_jobs[num_vars], median, na.rm = TRUE),
  Mode = sapply(cleaned_DS_jobs[num_vars], function(x) {
    table(factor(x, levels = unique(x)))[which.max(table(factor(x, levels = unique(x))))] 
  }),
  Min = sapply(cleaned_DS_jobs[num_vars], min, na.rm = TRUE),
  Max = sapply(cleaned_DS_jobs[num_vars], max, na.rm = TRUE),
  StdDev = sapply(cleaned_DS_jobs[num_vars], sd, na.rm = TRUE)
)

##EDA

# Analyzing the relationship between Rating and average salary
plot(cleaned_DS_jobs$Rating, cleaned_DS_jobs$avg_salary, xlab = "Rating", ylab = "Average Salary",
     main = "Relationship between Rating and Average Salary")

# Analyzing the distribution of job_state
barplot(table(cleaned_DS_jobs$job_state), xlab = "Job State", ylab = "Frequency",
        main = "Distribution of Job State")

# Analyzing the company_age variable
plot(cleaned_DS_jobs$company_age, cleaned_DS_jobs$avg_salary, xlab = "Company Age", ylab = "Average Salary",
     main = "Relationship between Company Age and Average Salary")

### Business Question and Data Story:
 ## Based on the descriptive statistics and data exploration, we can formulate specific business questions and uncover the story the data is telling. For example:
  
  #How does the average salary vary across different job roles and sectors?
 # Is there a correlation between company rating and average salary?
 # Which states have the highest demand for data-related job roles?
 # Are certain industries more likely to offer higher salaries for data-related positions?
 # Does the age of a company impact the average salary it offers for data-related roles?
  
# Boxplot to identify outliers
boxplot(cleaned_DS_jobs$avg_salary, main = "Boxplot of Average Salary", ylab = "Average Salary")

# Calculate the IQR
Q1 <- quantile(cleaned_DS_jobs$avg_salary, 0.25)
Q3 <- quantile(cleaned_DS_jobs$avg_salary, 0.75)
IQR <- Q3 - Q1

# Determine the upper and lower bounds for outliers
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Identify outliers
outliers <- cleaned_DS_jobs$avg_salary < lower_bound | cleaned_DS_jobs$avg_salary > upper_bound

# Remove outliers
cleaned_DS_jobs <- cleaned_DS_jobs[!outliers, ]

# Boxplot for min_salary
boxplot(cleaned_DS_jobs$min_salary, main = "Boxplot of Minimum Salary", ylab = "Minimum Salary")

# Calculate the IQR for min_salary
Q1_min <- quantile(cleaned_DS_jobs$min_salary, 0.25)
Q3_min <- quantile(cleaned_DS_jobs$min_salary, 0.75)
IQR_min <- Q3_min - Q1_min

# Determine the upper and lower bounds for min_salary outliers
lower_bound_min <- Q1_min - 1.5 * IQR_min
upper_bound_min <- Q3_min + 1.5 * IQR_min

# Identify min_salary outliers
outliers_min <- cleaned_DS_jobs$min_salary < lower_bound_min | cleaned_DS_jobs$min_salary > upper_bound_min

# Remove min_salary outliers
cleaned_DS_jobs <- cleaned_DS_jobs[!outliers_min, ]


# Boxplot for max_salary
boxplot(cleaned_DS_jobs$max_salary, main = "Boxplot of Maximum Salary", ylab = "Maximum Salary")

# Calculate the IQR for max_salary
Q1_max <- quantile(cleaned_DS_jobs$max_salary, 0.25)
Q3_max <- quantile(cleaned_DS_jobs$max_salary, 0.75)
IQR_max <- Q3_max - Q1_max

# Determine the upper and lower bounds for max_salary outliers
lower_bound_max <- Q1_max - 1.5 * IQR_max
upper_bound_max <- Q3_max + 1.5 * IQR_max

# Identify max_salary outliers
outliers_max <- cleaned_DS_jobs$max_salary < lower_bound_max | cleaned_DS_jobs$max_salary > upper_bound_max

# Remove max_salary outliers
cleaned_DS_jobs <- cleaned_DS_jobs[!outliers_max, ]


# Boxplot for salary_diff
boxplot(cleaned_DS_jobs$salary_diff, main = "Boxplot of Salary Difference", ylab = "Salary Difference")

# Calculate the IQR for salary_diff
Q1_diff <- quantile(cleaned_DS_jobs$salary_diff, 0.25)
Q3_diff <- quantile(cleaned_DS_jobs$salary_diff, 0.75)
IQR_diff <- Q3_diff - Q1_diff

# Determine the upper and lower bounds for salary_diff outliers
lower_bound_diff <- Q1_diff - 1.5 * IQR_diff
upper_bound_diff <- Q3_diff + 1.5 * IQR_diff

# Identify salary_diff outliers
outliers_diff <- cleaned_DS_jobs$salary_diff < lower_bound_diff | cleaned_DS_jobs$salary_diff > upper_bound_diff

# Remove salary_diff outliers
cleaned_DS_jobs <- cleaned_DS_jobs[!outliers_diff, ]


# Compute frequency distribution of Revenue
revenue_freq <- table(cleaned_DS_jobs $Revenue)

# Sort the frequency table in descending order
revenue_freq <- sort(revenue_freq, decreasing = TRUE)

# Print the frequency table
print(revenue_freq)

library(dplyr)

# Remove last 6 characters from Revenue column
cleaned_DS_jobs$Revenue <- substr(cleaned_DS_jobs$Revenue, 1, nchar(cleaned_DS_jobs$Revenue) - 6)

# Replace specific values with "Not Available"
cleaned_DS_jobs$Revenue <- ifelse(cleaned_DS_jobs$Revenue %in% c("Unknown / Non-Applicable", "-1", "", "Unknown / Non-Appl"),
                     "Not Available", cleaned_DS_jobs$Revenue)

library(plotly)

# Count the number of companies for each revenue category
revenue_counts <- table(cleaned_DS_jobs$Revenue)

# Create a bar plot
fig <- plot_ly(x = names(revenue_counts), y = revenue_counts, type = "bar",
               marker = list(color = "green", line = list(color = "rgb(8,48,107)", width = 1.5)),
               opacity = 0.6)

# Add title and axis labels
fig <- fig %>% layout(title = "Number of Companies by Revenue",
                      xaxis = list(title = "Revenue"),
                      yaxis = list(title = "Number of Companies"))

# Display the plot
fig


cleaned_DS_jobs$Size <- ifelse(cleaned_DS_jobs$Size == "-1", "Unknown", cleaned_DS_jobs$Size)


size_filter <- function(size) {
  y <- gsub("\\s+[a-z]+", "", size)
  x <- gsub("\\s", "-", y)
  return(x)
}

library(dplyr)

cleaned_DS_jobs <- cleaned_DS_jobs %>% mutate(Size = size_filter(Size))

unique_sizes <- unique(cleaned_DS_jobs$Size)
print(unique_sizes)

library(plotly)

# Create data for the pie chart
labels <- unique(cleaned_DS_jobs$Size)
values <- table(cleaned_DS_jobs$Size)

# Create the pie chart
fig <- plot_ly(labels = labels, values = values, type = "pie", textinfo = "label+percent", hole = 0.2)

# Customize the layout
fig <- fig %>% layout(title = "Number of Companies Sizes")

# Display the chart
fig



salary_categorizer <- function(salary) {
  salary_upper <- as.integer(sub(".*-", "", salary))
  
  if (salary_upper < 100) {
    return("<100")
  } else if (100 <= salary_upper && salary_upper < 150) {
    return("100-150")
  } else if (150 <= salary_upper && salary_upper < 200) {
    return("150-200")
  } else if (200 <= salary_upper && salary_upper < 250) {
    return("200-250")
  } else {
    return(">250")
  }
}



library(dplyr)

cleaned_DS_jobs <- cleaned_DS_jobs %>%
  mutate(`Salary Estimate` = sapply(`Salary Estimate`, salary_categorizer))

unique_salary_ranges <- unique(cleaned_DS_jobs$`Salary Estimate`)
print(unique_salary_ranges)



library(ggplot2)

# Create the bar plot
bar_plot <- ggplot(cleaned_DS_jobs, aes(x = `Salary Estimate`)) +
  geom_bar(fill = "blue") +
  labs(x = "Salary Range in Thousands", title = "Salary Ranges")

# Show the plot
print(bar_plot)

library(ggplot2)

# Create the histogram
histogram <- ggplot(cleaned_DS_jobs, aes(x = Rating)) +
  geom_histogram(fill = "blue", bins = 30) +
  labs(x = "Rating", y = "Count", title = "Rating Distribution")

# Show the histogram
print(histogram)

cleaned_DS_jobs$`Type of ownership` <- ifelse(cleaned_DS_jobs$`Type of ownership` == "-1", "Unknown", cleaned_DS_jobs$`Type of ownership`)


cleaned_DS_jobs$Industry <- ifelse(cleaned_DS_jobs$Industry == "-1", "Unknown", cleaned_DS_jobs$Industry)


# Calculate the frequency of values in the "Industry" column
industry_freq <- as.data.frame(table(cleaned_DS_jobs$Industry))

# Rename the columns
colnames(industry_freq) <- c("Industry", "Frequency")

# Sort the data frame by frequency in descending order
industry_freq <- industry_freq[order(industry_freq$Frequency, decreasing = TRUE), ]

# Reset the row names
row.names(industry_freq) <- NULL

# Print the resulting data frame
print(industry_freq)

top_industries <- head(sort(table(cleaned_DS_jobs$industry_freq), decreasing = TRUE), 10)


library(ggplot2)

ggplot(cleaned_DS_jobs, aes(x = job_state)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Jobs by State") +
  xlab("State") +
  ylab("Count") +
  theme_bw()

# Subset the top 10 industries


# Compute mean
mean_avg_salary <- mean(cleaned_DS_jobs$avg_salary)
mean_max_salary <- mean(cleaned_DS_jobs$max_salary)
mean_min_salary <- mean(cleaned_DS_jobs$min_salary)
mean_salary_diff <- mean(cleaned_DS_jobs$salary_diff)

# Compute median
median_avg_salary <- median(cleaned_DS_jobs$avg_salary)
median_max_salary <- median(cleaned_DS_jobs$max_salary)
median_min_salary <- median(cleaned_DS_jobs$min_salary)
median_salary_diff <- median(cleaned_DS_jobs$salary_diff)


