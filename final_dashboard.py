import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk  # Add this import for ttk widgets including Notebook
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# Load YOLO model
model = YOLO("yolo11n.pt")  # Ensure correct model path

# Class names for detection
CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    7: "Truck"
}

# Traffic classification function
def classify_traffic(vehicle_count):
    """Classify traffic based on vehicle count"""
    if vehicle_count < 5:
        return "LOW", (0, 255, 0)  # Green color
    elif vehicle_count < 15:
        return "MEDIUM", (0, 165, 255)  # Orange color
    else:
        return "HIGH", (0, 0, 255)  # Red color

# Video paths for day and night
day_video_paths = ["DAY1.mp4", "DAY2.mp4", "DAY3.mp4", "DAY4.mp4"]
night_video_paths = ["NIGHT1.mp4", "NIGHT2.mp4", "NIGHT3.mp4", "NIGHT4.mp4"]

# Global variable to store current video paths
current_video_paths = day_video_paths

# Open video captures
caps = [cv2.VideoCapture(path) for path in current_video_paths]

# Check if videos opened successfully
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Unable to open video file {current_video_paths[i]}")
        exit(1)

# Function to count vehicles in the first frame
def count_vehicles(cap):
    ret, frame = cap.read()
    if not ret:
        return 0
    results = model(frame)
    count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in CLASS_NAMES:
                count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning
    return count

# Count vehicles in the first frame for each video
vehicle_counts = [count_vehicles(cap) for cap in caps]

# Calculate green signal durations based on vehicle counts
green_times = [count * 1.5 for count in vehicle_counts]  # Green time = vehicle count * 1.5 seconds

# Use the highest green time for L1/L3 and L2/L4 pairs
green_times[0] = green_times[2] = max(green_times[0], green_times[2])  # L1 and L3
green_times[1] = green_times[3] = max(green_times[1], green_times[3])  # L2 and L4

# Initialize red_times based on green_times
red_times = [0] * 4  # Initialize red_times with zeros
red_times[1] = red_times[3] = green_times[0]  # Red time for L2/L4 = Green time for L1/L3
red_times[0] = red_times[2] = green_times[1]  # Red time for L1/L3 = Green time for L2/L4

# Data storage for dashboard
traffic_history = {
    'lane1': {'time': [], 'count': [], 'vehicle_types': {}, 'traffic_level': []},
    'lane2': {'time': [], 'count': [], 'vehicle_types': {}, 'traffic_level': []},
    'lane3': {'time': [], 'count': [], 'vehicle_types': {}, 'traffic_level': []},
    'lane4': {'time': [], 'count': [], 'vehicle_types': {}, 'traffic_level': []}
}

start_time = time.time()

# Function to update traffic history
def update_traffic_history(lane_index, count, vehicle_types=None):
    lane_key = f'lane{lane_index+1}'
    current_time = time.time() - start_time
    
    traffic_history[lane_key]['time'].append(current_time)
    traffic_history[lane_key]['count'].append(count)
    
    # Add traffic classification
    traffic_level, _ = classify_traffic(count)
    traffic_history[lane_key]['traffic_level'].append(traffic_level)
    
    # Keep only the last 100 data points to prevent memory issues
    if len(traffic_history[lane_key]['time']) > 100:
        traffic_history[lane_key]['time'].pop(0)
        traffic_history[lane_key]['count'].pop(0)
        traffic_history[lane_key]['traffic_level'].pop(0)
    
    # Update vehicle types if provided
    if vehicle_types:
        for vehicle_type, type_count in vehicle_types.items():
            if vehicle_type not in traffic_history[lane_key]['vehicle_types']:
                traffic_history[lane_key]['vehicle_types'][vehicle_type] = []
            
            traffic_history[lane_key]['vehicle_types'][vehicle_type].append(type_count)
            
            # Keep only the last 100 data points
            if len(traffic_history[lane_key]['vehicle_types'][vehicle_type]) > 100:
                traffic_history[lane_key]['vehicle_types'][vehicle_type].pop(0)

# Function to overlay signal, timer, vehicle count, and traffic classification
def overlay_signal(frame, signal_status, timer, vehicle_count, lane_number):
    color = (0, 255, 0) if signal_status == "GO!" else (0, 0, 255)
    
    # Get traffic classification
    traffic_level, traffic_color = classify_traffic(vehicle_count)
    
    # Main status line
    text = f"Lane {lane_number} | {signal_status} | {timer:.1f}s | Vehicles: {vehicle_count}"
    cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Traffic classification line
    traffic_text = f"Traffic Level: {traffic_level}"
    cv2.putText(frame, traffic_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, traffic_color, 2)
    
    return frame

# Function to perform object detection with bounding boxes
def detect_objects(frame):
    results = model(frame)
    count = 0
    vehicle_types = {"Car": 0, "Motorcycle": 0, "Truck": 0}
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in CLASS_NAMES:
                count += 1
                label = CLASS_NAMES[class_id]
                confidence = float(box.conf[0])
                
                # Update vehicle type count
                if label in vehicle_types:
                    vehicle_types[label] += 1
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence
                label_text = f"{label} {confidence:.2f}"
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
    
    return frame, count, vehicle_types

# Initialize Tkinter
root = tk.Tk()
root.title("Real-Time Traffic Monitoring System")
root.state('zoomed')  # Open the window in maximized state

# Function to switch video paths to day mode
def switch_to_day():
    global current_video_paths, caps, last_frames, vehicle_counts, green_times, red_times
    current_video_paths = day_video_paths
    for cap in caps:
        cap.release()  # Release existing video captures
    caps = [cv2.VideoCapture(path) for path in current_video_paths]
    last_frames = [None] * 4  # Reset last_frames
    vehicle_counts = [count_vehicles(cap) for cap in caps]  # Re-count vehicles
    green_times = [count * 1.5 for count in vehicle_counts]  # Recalculate green times
    green_times[0] = green_times[2] = max(green_times[0], green_times[2])  # L1 and L3
    green_times[1] = green_times[3] = max(green_times[1], green_times[3])  # L2 and L4
    red_times[1] = red_times[3] = green_times[0]  # Red time for L2/L4 = Green time for L1/L3
    red_times[0] = red_times[2] = green_times[1]  # Red time for L1/L3 = Green time for L2/L4
    print("Switched to Day mode")

# Function to switch video paths to night mode
def switch_to_night():
    global current_video_paths, caps, last_frames, vehicle_counts, green_times, red_times
    current_video_paths = night_video_paths
    for cap in caps:
        cap.release()  # Release existing video captures
    caps = [cv2.VideoCapture(path) for path in current_video_paths]
    last_frames = [None] * 4  # Reset last_frames
    vehicle_counts = [count_vehicles(cap) for cap in caps]  # Re-count vehicles
    green_times = [count * 1.5 for count in vehicle_counts]  # Recalculate green times
    green_times[0] = green_times[2] = max(green_times[0], green_times[2])  # L1 and L3
    green_times[1] = green_times[3] = max(green_times[1], green_times[3])  # L2 and L4
    red_times[1] = red_times[3] = green_times[0]  # Red time for L2/L4 = Green time for L1/L3
    red_times[0] = red_times[2] = green_times[1]  # Red time for L1/L3 = Green time for L2/L4
    print("Switched to Night mode")

# Dashboard window
dashboard_window = None

# Function to create and show the dashboard
def show_dashboard():
    global dashboard_window
    
    # If dashboard is already open, bring it to front
    if dashboard_window is not None and dashboard_window.winfo_exists():
        dashboard_window.lift()
        return
    
    # Create new dashboard window
    dashboard_window = tk.Toplevel(root)
    dashboard_window.title("Real-Time Traffic Monitoring Dashboard - Analytics & Insights")
    dashboard_window.geometry("1400x900")
    dashboard_window.state('zoomed')  # Maximize dashboard
    
    # Create notebook (tabs)
    notebook = ttk.Notebook(dashboard_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ========== TAB 1: OVERVIEW DASHBOARD ==========
    tab_overview = tk.Frame(notebook, bg='#f0f0f0')
    notebook.add(tab_overview, text="📊 Overview Dashboard")
    
    # Top section - Key Metrics Cards
    metrics_frame = tk.Frame(tab_overview, bg='#f0f0f0')
    metrics_frame.pack(fill=tk.X, padx=20, pady=15)
    
    # Create 4 metric cards
    metric_cards = []
    metric_labels = {}
    
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#f44336']
    lane_names = ['Lane 1 (North)', 'Lane 2 (East)', 'Lane 3 (South)', 'Lane 4 (West)']
    
    for i in range(4):
        card = tk.Frame(metrics_frame, bg=colors[i], relief=tk.RAISED, borderwidth=3)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Lane name
        tk.Label(card, text=lane_names[i], font=("Arial", 14, "bold"), 
                bg=colors[i], fg='white').pack(pady=(10, 5))
        
        # Vehicle count (large)
        count_label = tk.Label(card, text="0", font=("Arial", 36, "bold"), 
                              bg=colors[i], fg='white')
        count_label.pack()
        metric_labels[f'count_{i}'] = count_label
        
        tk.Label(card, text="Vehicles", font=("Arial", 10), 
                bg=colors[i], fg='white').pack()
        
        # Traffic level
        level_label = tk.Label(card, text="LOW", font=("Arial", 12, "bold"), 
                              bg=colors[i], fg='#FFEB3B')
        level_label.pack(pady=5)
        metric_labels[f'level_{i}'] = level_label
        
        # Signal status
        signal_label = tk.Label(card, text="🔴 STOP", font=("Arial", 11), 
                               bg=colors[i], fg='white')
        signal_label.pack(pady=(0, 10))
        metric_labels[f'signal_{i}'] = signal_label
        
        metric_cards.append(card)
    
    # Middle section - Live Traffic Graphs
    graphs_frame = tk.Frame(tab_overview, bg='white', relief=tk.SUNKEN, borderwidth=2)
    graphs_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    tk.Label(graphs_frame, text="📈 Real-Time Traffic Flow", 
            font=("Arial", 16, "bold"), bg='white').pack(pady=10)
    
    fig_overview = plt.Figure(figsize=(14, 5), dpi=100, facecolor='white')
    
    # Left plot - Traffic volume over time
    ax_traffic = fig_overview.add_subplot(121)
    
    # Right plot - Current comparison
    ax_compare = fig_overview.add_subplot(122)
    
    canvas_overview = FigureCanvasTkAgg(fig_overview, graphs_frame)
    canvas_overview.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Bottom section - Statistics Table
    stats_table_frame = tk.Frame(tab_overview, bg='white', relief=tk.SUNKEN, borderwidth=2)
    stats_table_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
    
    tk.Label(stats_table_frame, text="📋 Detailed Statistics", 
            font=("Arial", 16, "bold"), bg='white').pack(pady=10)
    
    # Create table
    table_frame = tk.Frame(stats_table_frame, bg='white')
    table_frame.pack(padx=20, pady=(0, 20))
    
    headers = ["Lane", "Current", "Average", "Peak", "Min", "Traffic Level", "Signal", "Wait Time"]
    header_widths = [15, 10, 10, 10, 10, 15, 10, 12]
    
    for col, (header, width) in enumerate(zip(headers, header_widths)):
        tk.Label(table_frame, text=header, font=("Arial", 11, "bold"), 
                bg='#2196F3', fg='white', width=width, relief=tk.RAISED, 
                borderwidth=1).grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
    
    # Create data rows
    table_labels = {}
    for row in range(4):
        for col, width in enumerate(header_widths):
            label = tk.Label(table_frame, text="0" if col > 0 else lane_names[row], 
                           font=("Arial", 10), bg='white', width=width, 
                           relief=tk.RAISED, borderwidth=1)
            label.grid(row=row+1, column=col, sticky="nsew", padx=1, pady=1)
            table_labels[f'row{row}_col{col}'] = label
    
    # ========== TAB 2: VEHICLE ANALYSIS ==========
    tab2 = tk.Frame(notebook, bg='#f0f0f0')
    notebook.add(tab2, text="🚗 Vehicle Analysis")
    
    tk.Label(tab2, text="Vehicle Type Distribution by Lane", 
            font=("Arial", 18, "bold"), bg='#f0f0f0').pack(pady=15)
    
    # Create 2x2 grid for pie charts with better layout
    pie_container = tk.Frame(tab2, bg='white', relief=tk.SUNKEN, borderwidth=2)
    pie_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Create 2x2 grid for vehicle distribution per lane
    fig2 = plt.Figure(figsize=(12, 8), dpi=100, facecolor='white')
    ax2_1 = fig2.add_subplot(221)
    ax2_2 = fig2.add_subplot(222)
    ax2_3 = fig2.add_subplot(223)
    ax2_4 = fig2.add_subplot(224)
    canvas2 = FigureCanvasTkAgg(fig2, pie_container)
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Summary section
    summary_frame = tk.Frame(tab2, bg='white', relief=tk.SUNKEN, borderwidth=2)
    summary_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
    
    tk.Label(summary_frame, text="📊 Total Vehicle Summary", 
            font=("Arial", 14, "bold"), bg='white').pack(pady=10)
    
    summary_labels = {}
    summary_stats = tk.Frame(summary_frame, bg='white')
    summary_stats.pack(pady=10)
    
    vehicle_types_summary = ['🚗 Cars', '🏍️ Motorcycles', '🚛 Trucks', '📈 Total']
    summary_colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    
    for i, (v_type, color) in enumerate(zip(vehicle_types_summary, summary_colors)):
        frame = tk.Frame(summary_stats, bg=color, relief=tk.RAISED, borderwidth=2)
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Label(frame, text=v_type, font=("Arial", 12, "bold"), 
                bg=color, fg='white').pack(padx=20, pady=5)
        
        count_label = tk.Label(frame, text="0", font=("Arial", 24, "bold"), 
                              bg=color, fg='white')
        count_label.pack(padx=20, pady=10)
        summary_labels[v_type] = count_label
    
    # ========== TAB 3: TRAFFIC TRENDS ==========
    tab3 = tk.Frame(notebook, bg='#f0f0f0')
    notebook.add(tab3, text="📈 Traffic Trends")
    
    tk.Label(tab3, text="Historical Traffic Analysis", 
            font=("Arial", 18, "bold"), bg='#f0f0f0').pack(pady=15)
    
    # Traffic timeline graph
    timeline_frame = tk.Frame(tab3, bg='white', relief=tk.SUNKEN, borderwidth=2)
    timeline_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    fig_trends = plt.Figure(figsize=(12, 8), dpi=100, facecolor='white')
    
    # Top plot - Traffic volume over time
    ax_timeline = fig_trends.add_subplot(211)
    
    # Bottom plot - Traffic levels over time
    ax_levels = fig_trends.add_subplot(212)
    
    canvas_trends = FigureCanvasTkAgg(fig_trends, timeline_frame)
    canvas_trends.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ========== TAB 4: PERFORMANCE METRICS ==========
    tab4 = tk.Frame(notebook, bg='#f0f0f0')
    notebook.add(tab4, text="⚡ Performance")
    
    tk.Label(tab4, text="System Performance Metrics", 
            font=("Arial", 18, "bold"), bg='#f0f0f0').pack(pady=15)
    
    # Performance cards
    perf_frame = tk.Frame(tab4, bg='#f0f0f0')
    perf_frame.pack(fill=tk.X, padx=20, pady=20)
    
    perf_metrics = [
        ("⏱️ Total Runtime", "0s", "#2196F3"),
        ("🚦 Signal Cycles", "0", "#4CAF50"),
        ("📊 Avg Wait Time", "0s", "#FF9800"),
        ("🎯 Detection Rate", "0%", "#9C27B0")
    ]
    
    perf_labels = {}
    for title, value, color in perf_metrics:
        card = tk.Frame(perf_frame, bg=color, relief=tk.RAISED, borderwidth=3)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(card, text=title, font=("Arial", 12, "bold"), 
                bg=color, fg='white').pack(pady=10)
        
        value_label = tk.Label(card, text=value, font=("Arial", 28, "bold"), 
                              bg=color, fg='white')
        value_label.pack(pady=20)
        perf_labels[title] = value_label
    
    # Efficiency graph
    efficiency_frame = tk.Frame(tab4, bg='white', relief=tk.SUNKEN, borderwidth=2)
    efficiency_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    tk.Label(efficiency_frame, text="Traffic Flow Efficiency", 
            font=("Arial", 14, "bold"), bg='white').pack(pady=10)
    
    fig_perf = plt.Figure(figsize=(12, 6), dpi=100, facecolor='white')
    ax_perf = fig_perf.add_subplot(111)
    canvas_perf = FigureCanvasTkAgg(fig_perf, efficiency_frame)
    canvas_perf.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Function to update dashboard
    signal_cycle_count = 0
    def update_dashboard():
        nonlocal signal_cycle_count
        if not dashboard_window.winfo_exists():
            return
        
        # ===== UPDATE OVERVIEW TAB =====
        # Update metric cards
        for i in range(4):
            lane_key = f'lane{i+1}'
            if traffic_history[lane_key]['count']:
                current_count = traffic_history[lane_key]['count'][-1]
                metric_labels[f'count_{i}'].config(text=str(current_count))
                
                # Update traffic level
                current_level = traffic_history[lane_key]['traffic_level'][-1] if traffic_history[lane_key]['traffic_level'] else "LOW"
                level_color = '#FFEB3B' if current_level == "LOW" else '#FFC107' if current_level == "MEDIUM" else '#FF5252'
                metric_labels[f'level_{i}'].config(text=current_level, fg=level_color)
                
                # Update signal status
                signal_text = "🟢 GO!" if signals[i] == "GO!" else "🔴 STOP"
                metric_labels[f'signal_{i}'].config(text=signal_text)
        
        # Update traffic flow graph
        ax_traffic.clear()
        for i, lane_key in enumerate(traffic_history.keys()):
            if traffic_history[lane_key]['time'] and traffic_history[lane_key]['count']:
                ax_traffic.plot(
                    traffic_history[lane_key]['time'], 
                    traffic_history[lane_key]['count'],
                    label=lane_names[i],
                    linewidth=2,
                    marker='o',
                    markersize=3
                )
        
        ax_traffic.set_title("Traffic Volume Over Time", fontsize=14, fontweight='bold')
        ax_traffic.set_xlabel("Time (seconds)", fontsize=11)
        ax_traffic.set_ylabel("Vehicle Count", fontsize=11)
        ax_traffic.legend(loc='upper left', fontsize=9)
        ax_traffic.grid(True, alpha=0.3)
        ax_traffic.set_facecolor('#f9f9f9')
        
        # Update comparison bar chart
        ax_compare.clear()
        current_counts = []
        for i, lane_key in enumerate(traffic_history.keys()):
            if traffic_history[lane_key]['count']:
                current_counts.append(traffic_history[lane_key]['count'][-1])
            else:
                current_counts.append(0)
        
        bars = ax_compare.bar(range(4), current_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax_compare.set_title("Current Lane Comparison", fontsize=14, fontweight='bold')
        ax_compare.set_xlabel("Lanes", fontsize=11)
        ax_compare.set_ylabel("Vehicles", fontsize=11)
        ax_compare.set_xticks(range(4))
        ax_compare.set_xticklabels(['L1', 'L2', 'L3', 'L4'])
        ax_compare.grid(True, axis='y', alpha=0.3)
        ax_compare.set_facecolor('#f9f9f9')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_compare.text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}',
                          ha='center', va='bottom', fontweight='bold')
        
        canvas_overview.draw()
        
        # Update statistics table
        for i, lane_key in enumerate(traffic_history.keys()):
            if traffic_history[lane_key]['count']:
                current_count = traffic_history[lane_key]['count'][-1]
                avg_count = sum(traffic_history[lane_key]['count']) / len(traffic_history[lane_key]['count'])
                peak_count = max(traffic_history[lane_key]['count'])
                min_count = min(traffic_history[lane_key]['count'])
                
                table_labels[f'row{i}_col1'].config(text=str(current_count))
                table_labels[f'row{i}_col2'].config(text=f"{avg_count:.1f}")
                table_labels[f'row{i}_col3'].config(text=str(peak_count))
                table_labels[f'row{i}_col4'].config(text=str(min_count))
                
                current_level = traffic_history[lane_key]['traffic_level'][-1] if traffic_history[lane_key]['traffic_level'] else "LOW"
                level_color = "green" if current_level == "LOW" else "orange" if current_level == "MEDIUM" else "red"
                table_labels[f'row{i}_col5'].config(text=current_level, fg=level_color, font=("Arial", 10, "bold"))
                
                signal_text = "🟢 GO" if signals[i] == "GO!" else "🔴 STOP"
                table_labels[f'row{i}_col6'].config(text=signal_text)
                
                wait_time = f"{red_times[i]:.1f}s" if signals[i] == "STOP" else "0s"
                table_labels[f'row{i}_col7'].config(text=wait_time)
        
        # ===== UPDATE VEHICLE ANALYSIS TAB =====
        ax2_1.clear()
        ax2_2.clear()
        ax2_3.clear()
        ax2_4.clear()
        
        axes = [ax2_1, ax2_2, ax2_3, ax2_4]
        total_cars = total_motorcycles = total_trucks = 0
        
        for i, lane_key in enumerate(traffic_history.keys()):
            vehicle_data = traffic_history[lane_key]['vehicle_types']
            vehicle_counts_dict = {}
            
            for v_type in vehicle_data:
                if vehicle_data[v_type]:
                    count = vehicle_data[v_type][-1]
                    vehicle_counts_dict[v_type] = count
                    
                    # Sum totals
                    if v_type == "Car":
                        total_cars += count
                    elif v_type == "Motorcycle":
                        total_motorcycles += count
                    elif v_type == "Truck":
                        total_trucks += count
            
            if vehicle_counts_dict:
                wedges, texts, autotexts = axes[i].pie(
                    vehicle_counts_dict.values(),
                    labels=vehicle_counts_dict.keys(),
                    autopct='%1.1f%%',
                    colors=['#4CAF50', '#2196F3', '#FF9800'],
                    startangle=90,
                    textprops={'fontsize': 10, 'fontweight': 'bold'}
                )
                axes[i].set_title(f"{lane_names[i]}\nTotal: {sum(vehicle_counts_dict.values())}", 
                                fontsize=11, fontweight='bold')
            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           fontsize=14, transform=axes[i].transAxes)
                axes[i].set_title(lane_names[i], fontsize=11, fontweight='bold')
        
        canvas2.draw()
        
        # Update summary
        summary_labels['🚗 Cars'].config(text=str(total_cars))
        summary_labels['🏍️ Motorcycles'].config(text=str(total_motorcycles))
        summary_labels['🚛 Trucks'].config(text=str(total_trucks))
        summary_labels['📈 Total'].config(text=str(total_cars + total_motorcycles + total_trucks))
        
        # ===== UPDATE TRAFFIC TRENDS TAB =====
        ax_timeline.clear()
        ax_levels.clear()
        
        # Plot traffic volume trends
        for i, lane_key in enumerate(traffic_history.keys()):
            if traffic_history[lane_key]['time'] and traffic_history[lane_key]['count']:
                ax_timeline.plot(
                    traffic_history[lane_key]['time'], 
                    traffic_history[lane_key]['count'],
                    label=lane_names[i],
                    linewidth=2.5,
                    marker='o',
                    markersize=4,
                    color=colors[i]
                )
        
        ax_timeline.set_title("Traffic Volume Timeline", fontsize=14, fontweight='bold')
        ax_timeline.set_xlabel("Time (seconds)", fontsize=11)
        ax_timeline.set_ylabel("Vehicle Count", fontsize=11)
        ax_timeline.legend(loc='best', fontsize=9)
        ax_timeline.grid(True, alpha=0.3)
        ax_timeline.set_facecolor('#f9f9f9')
        ax_timeline.fill_between([], [], alpha=0.2)
        
        # Plot traffic level distribution
        level_counts = {'LOW': [0]*4, 'MEDIUM': [0]*4, 'HIGH': [0]*4}
        for i, lane_key in enumerate(traffic_history.keys()):
            for level in traffic_history[lane_key]['traffic_level']:
                level_counts[level][i] += 1
        
        x = np.arange(4)
        width = 0.25
        
        ax_levels.bar(x - width, level_counts['LOW'], width, label='LOW', color='#4CAF50', alpha=0.8)
        ax_levels.bar(x, level_counts['MEDIUM'], width, label='MEDIUM', color='#FF9800', alpha=0.8)
        ax_levels.bar(x + width, level_counts['HIGH'], width, label='HIGH', color='#f44336', alpha=0.8)
        
        ax_levels.set_title("Traffic Level Distribution", fontsize=14, fontweight='bold')
        ax_levels.set_xlabel("Lanes", fontsize=11)
        ax_levels.set_ylabel("Occurrences", fontsize=11)
        ax_levels.set_xticks(x)
        ax_levels.set_xticklabels(['L1', 'L2', 'L3', 'L4'])
        ax_levels.legend(fontsize=9)
        ax_levels.grid(True, axis='y', alpha=0.3)
        ax_levels.set_facecolor('#f9f9f9')
        
        canvas_trends.draw()
        
        # ===== UPDATE PERFORMANCE TAB =====
        runtime = time.time() - start_time
        perf_labels["⏱️ Total Runtime"].config(text=f"{int(runtime)}s")
        
        # Count signal cycles (estimate)
        if runtime > 0:
            avg_cycle = (sum(green_times) + sum(red_times)) / 4 if sum(green_times) + sum(red_times) > 0 else 30
            signal_cycle_count = int(runtime / avg_cycle)
        perf_labels["🚦 Signal Cycles"].config(text=str(signal_cycle_count))
        
        # Average wait time
        total_wait = sum([red_times[i] if signals[i] == "STOP" else 0 for i in range(4)])
        avg_wait = total_wait / 4
        perf_labels["📊 Avg Wait Time"].config(text=f"{avg_wait:.1f}s")
        
        # Detection rate (vehicles per minute)
        total_vehicles = sum([traffic_history[f'lane{i+1}']['count'][-1] if traffic_history[f'lane{i+1}']['count'] else 0 for i in range(4)])
        detection_rate = (total_vehicles / runtime * 60) if runtime > 0 else 0
        perf_labels["🎯 Detection Rate"].config(text=f"{detection_rate:.1f}/min")
        
        # Efficiency graph - throughput over time
        ax_perf.clear()
        
        # Calculate throughput (vehicles per time window)
        window_size = 10  # 10 second windows
        for i, lane_key in enumerate(traffic_history.keys()):
            if len(traffic_history[lane_key]['time']) > 1:
                times = traffic_history[lane_key]['time']
                counts = traffic_history[lane_key]['count']
                
                throughput = []
                time_windows = []
                
                for j in range(0, len(times), window_size):
                    if j + window_size < len(times):
                        window_throughput = counts[j + window_size] - counts[j]
                        throughput.append(window_throughput)
                        time_windows.append(times[j + window_size])
                
                if throughput:
                    ax_perf.plot(time_windows, throughput, label=lane_names[i], 
                               linewidth=2, marker='s', markersize=5, color=colors[i])
        
        ax_perf.set_title("Traffic Throughput (Vehicles per 10s window)", fontsize=14, fontweight='bold')
        ax_perf.set_xlabel("Time (seconds)", fontsize=11)
        ax_perf.set_ylabel("Throughput", fontsize=11)
        ax_perf.legend(loc='best', fontsize=9)
        ax_perf.grid(True, alpha=0.3)
        ax_perf.set_facecolor('#f9f9f9')
        
        canvas_perf.draw()
        
        # Schedule next update
        dashboard_window.after(500, update_dashboard)
    
    # Start dashboard updates
    update_dashboard()

# Load icons
day_icon = tk.PhotoImage(file="day.png")
night_icon = tk.PhotoImage(file="night.png")
try:
    dashboard_icon = tk.PhotoImage(file="dashboard.png")
except:
    # Create a simple dashboard icon if the file doesn't exist
    dashboard_icon = tk.PhotoImage(width=32, height=32)
    dashboard_icon.put("blue", (0, 0, 31, 31))
    dashboard_icon.put("white", (3, 3, 28, 7))
    dashboard_icon.put("white", (3, 10, 28, 14))
    dashboard_icon.put("white", (3, 17, 28, 21))
    dashboard_icon.put("white", (3, 24, 28, 28))

# Create buttons for day and night mode, and dashboard
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, padx=15, pady=5)

day_button = tk.Button(button_frame, image=day_icon, command=switch_to_day)
day_button.pack(side=tk.LEFT, padx=5)

night_button = tk.Button(button_frame, image=night_icon, command=switch_to_night)
night_button.pack(side=tk.LEFT, padx=5)

dashboard_button = tk.Button(button_frame, image=dashboard_icon, command=show_dashboard)
dashboard_button.pack(side=tk.LEFT, padx=5)
tk.Label(button_frame, text="Show Dashboard", font=("Arial", 10)).pack(side=tk.LEFT)

# Create a container for video frames
frame_container = tk.Frame(root)
frame_container.pack(fill=tk.BOTH, expand=True)

# Configure grid to expand and fill the container
for i in range(2):
    frame_container.grid_columnconfigure(i, weight=1, uniform="column")
    frame_container.grid_rowconfigure(i, weight=1, uniform="row")

# Create labels for each video feed
labels = []
for i in range(4):
    label = tk.Label(frame_container)
    label.grid(row=i // 2, column=i % 2, sticky="nsew", padx=3, pady=3)  # Arrange in 2x2 grid
    labels.append(label)

# Store the last frame for each video when the signal is red
last_frames = [None] * 4

# Track signal status
signals = ["GO!", "STOP", "GO!", "STOP"]  # Initial signal states

# Function to update video frames in Tkinter
def update_frames():
    global signals
    frames = []

    # Update signals based on green and red times
    for i in range(4):
        if green_times[i] > 0:
            signals[i] = "GO!"
        else:
            signals[i] = "STOP"

    # Ensure L1/L3 and L2/L4 have opposite signals
    if signals[0] == "GO!" or signals[2] == "GO!":
        signals[1] = "STOP"
        signals[3] = "STOP"
    elif signals[1] == "GO!" or signals[3] == "GO!":
        signals[0] = "STOP"
        signals[2] = "STOP"

    for i, cap in enumerate(caps):
        if signals[i] == "STOP":  
            # STOP: Pause the video and show the last frame with detection
            if last_frames[i] is not None:
                frame = last_frames[i].copy()
                # Perform detection on stopped frame
                frame, count, vehicle_types = detect_objects(frame)
                vehicle_counts[i] = count
                update_traffic_history(i, count, vehicle_types)
            else:
                # If no last frame is available, read one frame and pause
                ret, frame = cap.read()
                if ret:
                    last_frames[i] = frame.copy()
                    frame, count, vehicle_types = detect_objects(frame)
                    vehicle_counts[i] = count
                    update_traffic_history(i, count, vehicle_types)
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Default black frame
        else:
            # GO: Read frame for lanes with green signals and perform live detection
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            # Store original frame
            last_frames[i] = frame.copy()
            
            # Perform live detection on every frame
            frame, count, vehicle_types = detect_objects(frame)
            vehicle_counts[i] = count
            update_traffic_history(i, count, vehicle_types)
        
        # Resize frame dynamically to fit the label size
        label_width = labels[i].winfo_width()
        label_height = labels[i].winfo_height()
        if label_width > 0 and label_height > 0:  # Ensure label size is valid
            frame = cv2.resize(frame, (label_width, label_height))
        else:
            frame = cv2.resize(frame, (640, 480))  # Default size if label size is invalid

        # Overlay signal, timer, vehicle count, and traffic classification
        frame = overlay_signal(frame, signals[i], green_times[i] if signals[i] == "GO!" else red_times[i], vehicle_counts[i], i + 1)
        frames.append(frame)
        
        # Update signal times (faster countdown)
        if signals[i] == "GO!":
            green_times[i] -= 0.5
        else:
            red_times[i] -= 0.5

        # Reset timers and switch signals when green time runs out
        if green_times[i] <= 0 and signals[i] == "GO!":
            green_times[i] = 0
            # Switch signals for opposite roads
            if i == 0 or i == 2:  # Road 1 or Road 3
                # Update green time for L2/L4 pair
                green_times[1] = green_times[3] = max(vehicle_counts[1] * 1.5, vehicle_counts[3] * 1.5)
                # Apply thresholds: minimum 10 seconds, maximum 120 seconds
                green_times[1] = green_times[3] = max(10, min(120, green_times[1]))
                red_times[0] = red_times[2] = green_times[1]  # Set red time for L1/L3
            elif i == 1 or i == 3:  # Road 2 or Road 4
                # Update green time for L1/L3 pair
                green_times[0] = green_times[2] = max(vehicle_counts[0] * 1.5, vehicle_counts[2] * 1.5)
                # Apply thresholds: minimum 10 seconds, maximum 120 seconds
                green_times[0] = green_times[2] = max(10, min(120, green_times[0]))
                red_times[1] = red_times[3] = green_times[0]  # Set red time for L2/L4

    # Convert frames to Tkinter-compatible images
    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        labels[i].config(image=imgtk)
        labels[i].image = imgtk  # Keep a reference to avoid garbage collection

    # Schedule the next update
    root.after(100, update_frames)

# Start the Tkinter main loop
update_frames()
root.mainloop()

# Release resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()