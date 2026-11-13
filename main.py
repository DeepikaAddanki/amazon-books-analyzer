import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/amazon_books.csv")

# Display the dataset
print(df)
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())
print("\nAverage Book Rating:")
print(df["rating"].mean())

print("\nHighest Book Price:")
print(df["price"].max())

print("\nMaximum Number of Reviews:")
print(df["reviews"].max())
# Find the index of the highest rating
best_index = df["rating"].idxmax()

# Retrieve the complete book details
best_book = df.loc[best_index]

print("\nHighest Rated Book:")
print(best_book)
top_books = df.sort_values(
    by="rating",
    ascending=False
).head(5)

print("\nTop 5 Highest Rated Books:")
print(top_books)
author_counts = df["author"].value_counts()

print("\nBooks by Each Author:")
print(author_counts)
genre_counts = df["genre"].value_counts()

print("\nBooks by Genre:")
print(genre_counts)
# Create a bar chart
plt.figure(figsize=(8, 5))

plt.bar(genre_counts.index, genre_counts.values)

plt.title("Number of Books by Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Books")

plt.savefig("visualizations/books_by_genre.png")

plt.show()
# Genre distribution pie chart
plt.figure(figsize=(7, 7))

plt.pie(
    genre_counts.values,
    labels=genre_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Genre Distribution of Books")

plt.savefig("visualizations/genre_distribution.png")

plt.show()
# Book price comparison
plt.figure(figsize=(10, 6))

plt.barh(df["title"], df["price"])

plt.title("Price Comparison of Books")
plt.xlabel("Price")
plt.ylabel("Book Title")

plt.tight_layout()

plt.savefig("visualizations/book_prices.png")

plt.show()
# Professional genre bar chart

plt.figure(figsize=(9, 5))

bars = plt.bar(
    genre_counts.index,
    genre_counts.values,
    color="skyblue",
    edgecolor="black"
)

plt.title("Amazon Books - Genre Analysis", fontsize=16)
plt.xlabel("Book Genre", fontsize=12)
plt.ylabel("Number of Books", fontsize=12)

# Display values above bars
plt.bar_label(bars)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig("visualizations/books_by_genre.png")

plt.show()
# Create a dashboard with 4 charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Books by genre
axes[0, 0].bar(genre_counts.index, genre_counts.values)
axes[0, 0].set_title("Books by Genre")
axes[0, 0].tick_params(axis="x", rotation=20)

# 2. Genre distribution
axes[0, 1].pie(
    genre_counts.values,
    labels=genre_counts.index,
    autopct="%1.1f%%"
)
axes[0, 1].set_title("Genre Distribution")

# 3. Ratings vs reviews
axes[1, 0].scatter(df["reviews"], df["rating"])
axes[1, 0].set_title("Ratings vs Reviews")
axes[1, 0].set_xlabel("Reviews")
axes[1, 0].set_ylabel("Rating")

# 4. Book prices
axes[1, 1].barh(df["title"], df["price"])
axes[1, 1].set_title("Book Price Comparison")

# Dashboard heading
fig.suptitle("Amazon Books Analysis Dashboard", fontsize=18)

plt.tight_layout()

plt.savefig("visualizations/books_dashboard.png")

plt.show()
# Average rating for each genre
genre_ratings = df.groupby("genre")["rating"].mean()

print("\nAverage Rating by Genre:")
print(genre_ratings)
# Average price of each genre
genre_prices = df.groupby("genre")["price"].mean()

print("\nAverage Price by Genre:")
print(genre_prices)
plt.figure(figsize=(8, 5))

plt.bar(
    genre_prices.index,
    genre_prices.values,
    color="orange"
)

plt.title("Average Book Price by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Price (₹)")

plt.tight_layout()

plt.savefig("visualizations/average_price_by_genre.png")

plt.show()
genre_analysis = df.groupby("genre").agg(
    average_rating=("rating", "mean"),
    average_price=("price", "mean"),
    total_reviews=("reviews", "sum")
)

print("\nComplete Genre Analysis:")
print(genre_analysis)
# Sort genres by total reviews
sorted_genres = genre_analysis.sort_values(
    by="total_reviews",
    ascending=False
)

print("\nGenres Sorted by Total Reviews:")
print(sorted_genres)
plt.figure(figsize=(9, 5))

bars = plt.bar(
    sorted_genres.index,
    sorted_genres["total_reviews"],
    color="mediumseagreen",
    edgecolor="black"
)

plt.title("Total Reviews by Book Genre", fontsize=16)
plt.xlabel("Genre")
plt.ylabel("Total Reviews")

plt.bar_label(bars)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig("visualizations/reviews_by_genre.png")

plt.show()
# Correlation between reviews and ratings
correlation = np.corrcoef(
    df["reviews"],
    df["rating"]
)[0, 1]

print("\nCorrelation between Reviews and Ratings:")
print(correlation)
# Correlation between all numerical columns
correlation_matrix = df[["rating", "price", "reviews"]].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)
# Correlation heatmap
plt.figure(figsize=(7, 5))

plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

# Display correlation values inside the chart
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j, i,
            round(correlation_matrix.iloc[i, j], 2),
            ha="center",
            va="center"
        )

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("visualizations/correlation_heatmap.png")

plt.show()
# Check missing values
missing_values = df.isnull().sum()

print("\nMissing Values:")
print(missing_values)
average_rating = df["rating"].mean()
print("Average Rating:", average_rating)
df["rating"] = df["rating"].fillna(average_rating)

print("\nMissing Ratings After Filling:")
print(df["rating"].isnull().sum())
df = df.drop_duplicates()

print("Rows after removing duplicates:", len(df))
# Classify books based on price

df["price_category"] = np.where(
    df["price"] < 300,
    "Low Price",
    np.where(
        df["price"] <= 450,
        "Medium Price",
        "High Price"
    )
)

print("\nBooks with Price Categories:")
print(df[["title", "price", "price_category"]])
# Count books in each price category
price_counts = df["price_category"].value_counts()

print("\nBooks by Price Category:")
print(price_counts)
plt.figure(figsize=(7, 7))

plt.pie(
    price_counts.values,
    labels=price_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    explode=[0.05] * len(price_counts)
)

plt.title("Book Price Category Distribution")

plt.tight_layout()

plt.savefig("visualizations/price_category_distribution.png")

plt.show()
# Calculate percentage of each price category
price_percentage = df["price_category"].value_counts(normalize=True) * 100

print("\nPrice Category Percentage:")
print(price_percentage)
for category, percentage in price_percentage.items():
    print(f"{category}: {percentage:.1f}%")
# Export analyzed data
df.to_csv("data/analyzed_books.csv", index=False)

print("\nAnalyzed data exported successfully!")
# Export genre analysis
genre_analysis.to_csv(
    "data/genre_analysis.csv"
)

print("Genre analysis exported successfully!")
# Read exported genre analysis
saved_genres = pd.read_csv("data/genre_analysis.csv")

print("\nExported Genre Analysis:")
print(saved_genres)
# Generate project insights

highest_rated = df.loc[df["rating"].idxmax()]
most_reviewed = df.loc[df["reviews"].idxmax()]
most_expensive = df.loc[df["price"].idxmax()]

print("\n========== PROJECT INSIGHTS ==========")

print("Highest Rated Book:", highest_rated["title"])
print("Rating:", highest_rated["rating"])

print("\nMost Reviewed Book:", most_reviewed["title"])
print("Reviews:", most_reviewed["reviews"])

print("\nMost Expensive Book:", most_expensive["title"])
print("Price: ₹", most_expensive["price"])

print("\nAverage Book Rating:", round(df["rating"].mean(), 2))
print("Average Book Price: ₹", round(df["price"].mean(), 2))
# Generate project report

with open("report.txt", "w", encoding="utf-8") as file:

    file.write("AMAZON BOOKS ANALYSIS REPORT\n")
    file.write("============================\n\n")

    file.write(f"Total Books: {len(df)}\n")
    file.write(f"Average Rating: {df['rating'].mean():.2f}\n")
    file.write(f"Average Price: ₹{df['price'].mean():.2f}\n\n")

    file.write("HIGHEST RATED BOOK\n")
    file.write(f"{highest_rated['title']} - {highest_rated['rating']}\n\n")

    file.write("MOST REVIEWED BOOK\n")
    file.write(f"{most_reviewed['title']} - {most_reviewed['reviews']} reviews\n\n")

    file.write("MOST EXPENSIVE BOOK\n")
    file.write(f"{most_expensive['title']} - ₹{most_expensive['price']}\n\n")

    file.write("GENRE ANALYSIS\n")
    file.write(genre_analysis.to_string())

print("Project report generated successfully!")
def display_dataset_info(df):
    print("\n========== DATASET INFORMATION ==========")

    print("Total Books:", len(df))
    print("Total Columns:", len(df.columns))

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nDataset Information:")
    print(df.info())
df = pd.read_csv("data/amazon_books.csv")

display_dataset_info(df)
def generate_insights(df):

    highest_rated = df.loc[df["rating"].idxmax()]
    most_reviewed = df.loc[df["reviews"].idxmax()]
    most_expensive = df.loc[df["price"].idxmax()]

    print("\n========== PROJECT INSIGHTS ==========")

    print("Highest Rated Book:", highest_rated["title"])
    print("Rating:", highest_rated["rating"])

    print("\nMost Reviewed Book:", most_reviewed["title"])
    print("Reviews:", most_reviewed["reviews"])

    print("\nMost Expensive Book:", most_expensive["title"])
    print("Price: ₹", most_expensive["price"])

    print("\nAverage Rating:", round(df["rating"].mean(), 2))
    print("Average Price:", round(df["price"].mean(), 2))
generate_insights(df)
def analyze_genres(df):

    genre_analysis = df.groupby("genre").agg(
        average_rating=("rating", "mean"),
        average_price=("price", "mean"),
        total_reviews=("reviews", "sum")
    )

    return genre_analysis
genre_analysis = analyze_genres(df)

print("\nGenre Analysis:")
print(genre_analysis)
def plot_genre_distribution(df):

    genre_counts = df["genre"].value_counts()

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        genre_counts.index,
        genre_counts.values,
        color="skyblue",
        edgecolor="black"
    )

    plt.title("Books by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Number of Books")

    plt.bar_label(bars)

    plt.tight_layout()

    plt.savefig("visualizations/books_by_genre.png")

    plt.show()
plot_genre_distribution(df)
def plot_price_comparison(df):

    plt.figure(figsize=(10, 6))

    plt.barh(df["title"], df["price"], color="orange")

    plt.title("Book Price Comparison")
    plt.xlabel("Price (₹)")
    plt.ylabel("Book Title")

    plt.tight_layout()

    plt.savefig("visualizations/book_prices.png")

    plt.show()
def plot_reviews_by_genre(df):

    genre_reviews = df.groupby("genre")["reviews"].sum()

    genre_reviews = genre_reviews.sort_values(
        ascending=False
    )

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        genre_reviews.index,
        genre_reviews.values,
        color="mediumseagreen",
        edgecolor="black"
    )

    plt.title("Total Reviews by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Total Reviews")

    plt.bar_label(bars)

    plt.tight_layout()

    plt.savefig("visualizations/reviews_by_genre.png")

    plt.show()
plot_reviews_by_genre(df)
def create_dashboard(df):

    genre_counts = df["genre"].value_counts()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Chart 1: Books by Genre
    axes[0, 0].bar(genre_counts.index, genre_counts.values)
    axes[0, 0].set_title("Books by Genre")
    axes[0, 0].tick_params(axis="x", rotation=20)

    # Chart 2: Genre Distribution
    axes[0, 1].pie(
        genre_counts.values,
        labels=genre_counts.index,
        autopct="%1.1f%%"
    )
    axes[0, 1].set_title("Genre Distribution")

    # Chart 3: Ratings vs Reviews
    axes[1, 0].scatter(df["reviews"], df["rating"])
    axes[1, 0].set_title("Ratings vs Reviews")
    axes[1, 0].set_xlabel("Reviews")
    axes[1, 0].set_ylabel("Rating")

    # Chart 4: Book Prices
    axes[1, 1].barh(df["title"], df["price"])
    axes[1, 1].set_title("Book Price Comparison")

    fig.suptitle("Amazon Books Analysis Dashboard", fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig("visualizations/books_dashboard.png")

    plt.show()
create_dashboard(df)
def main():

    # Load dataset
    df = pd.read_csv("data/amazon_books.csv")

    # Display dataset information
    display_dataset_info(df)

    # Generate insights
    generate_insights(df)

    # Genre analysis
    genre_analysis = analyze_genres(df)
    print("\nGenre Analysis:")
    print(genre_analysis)

    # Generate visualizations
    plot_genre_distribution(df)
    plot_price_comparison(df)
    plot_reviews_by_genre(df)
    create_dashboard(df)


if __name__ == "__main__":
    main()