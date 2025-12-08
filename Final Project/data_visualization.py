from project_imports import *
import json
import ast
from data_understanding import load_and_preprocess_data

def plot_trip_counts(train):
    """Plots the distribution of trip counts over time."""
    # Convert TIMESTAMP to datetime
    train['TIMESTAMP'] = pd.to_datetime(train['TIMESTAMP'], unit='s')
    
    # Extract date components
    train['hour'] = train['TIMESTAMP'].dt.hour
    train['day_of_week'] = train['TIMESTAMP'].dt.dayofweek
    train['month'] = train['TIMESTAMP'].dt.month
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.countplot(x='hour', data=train, ax=axes[0], palette='viridis')
    axes[0].set_title('Trips by Hour of Day')
    
    sns.countplot(x='day_of_week', data=train, ax=axes[1], palette='viridis')
    axes[1].set_title('Trips by Day of Week (0=Mon, 6=Sun)')
    
    sns.countplot(x='month', data=train, ax=axes[2], palette='viridis')
    axes[2].set_title('Trips by Month')
    
    plt.tight_layout()
    plt.savefig('trip_counts_distribution.png')
    print("Saved trip_counts_distribution.png")
    plt.close()

def plot_call_type_distribution(train):
    """Plots the distribution of call types."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='CALL_TYPE', data=train, palette='Set2')
    plt.title('Distribution of Call Types')
    plt.savefig('call_type_distribution.png')
    print("Saved call_type_distribution.png")
    plt.close()

def plot_trajectory_lengths(train):
    """Plots the distribution of trajectory lengths (number of points)."""
    # Parse POLYLINE if it's a string
    if isinstance(train['POLYLINE'].iloc[0], str):
        train['POLYLINE'] = train['POLYLINE'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
    train['num_points'] = train['POLYLINE'].apply(len)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(train['num_points'], bins=50, kde=True, color='blue')
    plt.title('Distribution of Trajectory Lengths (Number of Points)')
    plt.xlabel('Number of Points')
    plt.xlim(0, 500)  # Limit x-axis to focus on common lengths
    plt.savefig('trajectory_length_distribution.png')
    print("Saved trajectory_length_distribution.png")
    plt.close()

def plot_start_locations(train, sample_size=10000):
    """Plots a scatter map of trip start locations."""
    # Sample data for faster plotting
    sample_train = train.sample(n=min(sample_size, len(train)), random_state=42)
    
    # Parse POLYLINE if needed
    if isinstance(sample_train['POLYLINE'].iloc[0], str):
        sample_train['POLYLINE'] = sample_train['POLYLINE'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    # Extract start coordinates (first point)
    start_coords = []
    for poly in sample_train['POLYLINE']:
        if len(poly) > 0:
            start_coords.append(poly[0])
            
    start_df = pd.DataFrame(start_coords, columns=['lon', 'lat'])
    
    plt.figure(figsize=(10, 10))
    plt.scatter(start_df['lon'], start_df['lat'], s=1, alpha=0.1, color='red')
    plt.title(f'Start Locations of {sample_size} Sampled Trips')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # Set limits to focus on Porto area (approximate)
    plt.xlim(-8.75, -8.50)
    plt.ylim(41.10, 41.25)
    plt.savefig('start_locations_map.png')
    print("Saved start_locations_map.png")
    plt.close()

def plot_trajectories_sample(train, sample_size=100):
    """Plots a sample of full taxi trajectories."""
    # Parse POLYLINE if needed
    if isinstance(train['POLYLINE'].iloc[0], str):
        train['POLYLINE'] = train['POLYLINE'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
    sample_train = train.sample(n=min(sample_size, len(train)), random_state=42)
    
    plt.figure(figsize=(12, 10))
    
    # Plot each trajectory
    for poly in sample_train['POLYLINE']:
        if len(poly) > 1:
            poly = np.array(poly)
            plt.plot(poly[:, 0], poly[:, 1], linewidth=0.5, alpha=0.5, color='blue')
            
    plt.title(f'Sample of {sample_size} Taxi Trajectories')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # Focus on Porto
    plt.xlim(-8.70, -8.55)
    plt.ylim(41.10, 41.25)
    plt.grid(True, alpha=0.3)
    plt.savefig('trajectories_sample.png')
    print("Saved trajectories_sample.png")
    plt.close()

def plot_destinations_map(train, sample_size=50000):
    """Plots the distribution of trip destinations."""
    # Parse POLYLINE if needed
    if isinstance(train['POLYLINE'].iloc[0], str):
        train['POLYLINE'] = train['POLYLINE'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    # Extract last points
    dest_coords = []
    for poly in train['POLYLINE']:
        if len(poly) > 0:
            dest_coords.append(poly[-1])
            
    dest_df = pd.DataFrame(dest_coords, columns=['lon', 'lat'])
    
    # Sample if too large
    if len(dest_df) > sample_size:
        dest_df = dest_df.sample(n=sample_size, random_state=42)
        
    plt.figure(figsize=(12, 10))
    plt.scatter(dest_df['lon'], dest_df['lat'], s=1, alpha=0.1, color='green')
    plt.title(f'Map of Trip Destinations (n={len(dest_df)})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # Focus on Porto
    plt.xlim(-8.70, -8.55)
    plt.ylim(41.10, 41.25)
    plt.grid(True, alpha=0.3)
    plt.savefig('destinations_map.png')
    print("Saved destinations_map.png")
    plt.close()

def plot_correlation_heatmap(df, filename="correlation_heatmap.png"):
    """Plots a correlation heatmap for numeric features in the DataFrame."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

if __name__ == "__main__":
    print("Loading data...")
    train, test, sample, location = load_and_preprocess_data()
    
    print("Generating visualizations...")
    plot_trip_counts(train.copy())
    plot_call_type_distribution(train.copy())
    plot_trajectory_lengths(train.copy())
    plot_start_locations(train.copy())
    plot_trajectories_sample(train.copy())
    plot_destinations_map(train.copy())
    
    print("Visualization complete.")
