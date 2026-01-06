import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
from tkinterdnd2 import DND_FILES, TkinterDnD

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.dicom_loader import scan_directory
from src.core.ct_perfusion import calculate_ct_perfusion
from src.core.mri_perfusion import calculate_mri_perfusion

class PerfusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CT/MRI Perfusion Analysis")
        self.root.geometry("1400x900")
        
        # Setup Drag and Drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)
        
        # Loading State
        self.is_loading = False
        self.progress_var = tk.DoubleVar()
        
        # Data State
        self.loaded_series = []
        self.current_series = None
        self.pixel_data = None # 4D array (x, y, z, t) or 3D (x, y, t)
        
        # View State
        self.current_slice_idx = 0
        self.window_level = 50
        self.window_width = 100
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # ROI State
        self.aif_coords = None
        self.vof_coords = None
        self.roi_cid = None
        self.current_roi_type = None
        
        # Result ROI State
        self.res_roi_cid = None
        self.res_roi_coords = [] 
        self.is_measuring = False
        self.res_roi_patches = [] # To store drawn circles
        
        # Setup UI
        self.create_layout()
        
        # Bring to front
        self.root.lift()
        self.root.attributes('-topmost',True)
        self.root.after_idle(self.root.attributes,'-topmost',False)
        self.root.focus_force()
        
    def create_layout(self):
        # 1. Left Panel: File List
        self.left_panel = tk.Frame(self.root, width=250)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        tk.Label(self.left_panel, text="DICOM Series").pack(pady=5)
        self.btn_load = tk.Button(self.left_panel, text="Load Folder", command=self.load_folder)
        self.btn_load.pack(fill=tk.X, pady=5)
        
        self.listbox = tk.Listbox(self.left_panel)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.on_series_select)
        
        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.left_panel, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        tk.Label(self.left_panel, text="Drag & Drop Folders Here", fg="gray").pack(side=tk.BOTTOM, pady=5)
        
        # 2. Right Panel: Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Viewer
        self.tab_viewer = tk.Frame(self.notebook)
        self.notebook.add(self.tab_viewer, text="  Image Viewer  ")
        self.setup_viewer_tab()
        
        # Tab 2: TDC Analysis
        self.tab_tdc = tk.Frame(self.notebook)
        self.notebook.add(self.tab_tdc, text="  TDC Analysis  ")
        self.setup_tdc_tab()
        
        # Tab 3: Results Overview
        self.tab_results = tk.Frame(self.notebook)
        self.notebook.add(self.tab_results, text="  Results Overview  ")
        self.setup_results_tab()
        
    def setup_viewer_tab(self):
        # Layout: Center Image, Right/Bottom Controls
        
        # Controls Frame (Bottom)
        controls_frame = tk.Frame(self.tab_viewer)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Slice Slider
        tk.Label(controls_frame, text="Slice:").pack(side=tk.LEFT)
        self.slice_slider = tk.Scale(controls_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slice_change)
        self.slice_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Windowing
        tk.Label(controls_frame, text="WL:").pack(side=tk.LEFT, padx=5)
        self.wl_slider = tk.Scale(controls_frame, from_=-1000, to=3000, orient=tk.HORIZONTAL, command=self.on_window_change)
        self.wl_slider.set(50)
        self.wl_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Label(controls_frame, text="WW:").pack(side=tk.LEFT, padx=5)
        self.ww_slider = tk.Scale(controls_frame, from_=1, to=4000, orient=tk.HORIZONTAL, command=self.on_window_change)
        self.ww_slider.set(100)
        self.ww_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Buttons
        btn_frame = tk.Frame(self.tab_viewer)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
        
        self.btn_aif = tk.Button(btn_frame, text="ROI: AIF (Artery)", command=lambda: self.start_roi_selection('aif'), bg='#ffcccc')
        self.btn_aif.pack(side=tk.LEFT, padx=5)
        
        self.btn_vof = tk.Button(btn_frame, text="ROI: VOF (Vein)", command=lambda: self.start_roi_selection('vof'), bg='#ccccff')
        self.btn_vof.pack(side=tk.LEFT, padx=5)
        
        self.modality_var = tk.StringVar(value="CT")
        tk.Radiobutton(btn_frame, text="CT", variable=self.modality_var, value="CT").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(btn_frame, text="MRI", variable=self.modality_var, value="MRI").pack(side=tk.LEFT, padx=10)
        
        self.btn_calc = tk.Button(btn_frame, text="START CALCULATION", command=self.calculate, font=('Helvetica', 12, 'bold'), bg='#ccffcc')
        self.btn_calc.pack(side=tk.RIGHT, padx=20)
        
        # Image Canvas
        self.fig_viewer, self.ax_viewer = plt.subplots(figsize=(8, 8))
        self.fig_viewer.patch.set_facecolor('#f0f0f0')
        self.canvas_viewer = FigureCanvasTkAgg(self.fig_viewer, master=self.tab_viewer)
        self.canvas_viewer.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Bind Events
        # Scroll: Slice (Normal) / Zoom (Ctrl)
        self.canvas_viewer.mpl_connect('scroll_event', self.on_scroll)
        # Pan: Right Drag
        self.canvas_viewer.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas_viewer.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas_viewer.mpl_connect('motion_notify_event', self.on_mouse_drag)
        
        # Reset Button (Top Right Overlay or Toolbar? Let's put in controls)
        self.btn_reset = tk.Button(btn_frame, text="Reset View", command=self.reset_view)
        self.btn_reset.pack(side=tk.RIGHT, padx=5)
        
        self.btn_zoom_out = tk.Button(btn_frame, text="-", command=lambda: self.zoom(1.1), width=2)
        self.btn_zoom_out.pack(side=tk.RIGHT, padx=2)
        
        self.btn_zoom_in = tk.Button(btn_frame, text="+", command=lambda: self.zoom(1/1.1), width=2)
        self.btn_zoom_in.pack(side=tk.RIGHT, padx=2)
        
        # Instructions
        instr = tk.Label(self.tab_viewer, text="Controls: Left Click=ROI | Right Drag=Pan | Ctrl+Scroll=Zoom", fg="gray", font=("Arial", 9))
        instr.pack(side=tk.TOP, pady=2)

    def setup_tdc_tab(self):
        self.fig_tdc, self.ax_tdc = plt.subplots(figsize=(10, 6))
        self.canvas_tdc = FigureCanvasTkAgg(self.fig_tdc, master=self.tab_tdc)
        self.canvas_tdc.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def setup_results_tab(self):
        # Layout: Canvas (Top/Center), Controls (Bottom)
        
        # 1. Controls (Pack First to ensure visibility at bottom)
        res_ctrl = tk.Frame(self.tab_results)
        res_ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Measure Mode Button
        self.btn_measure = tk.Button(res_ctrl, text="Measure ROI", command=self.toggle_measure_mode, bg='#eeeeee')
        self.btn_measure.pack(side=tk.LEFT, padx=5)
        
        self.btn_clear_measure = tk.Button(res_ctrl, text="Clear ROI", command=self.clear_measure_roi)
        self.btn_clear_measure.pack(side=tk.LEFT, padx=5)
        
        self.lbl_res_val = tk.Label(res_ctrl, text="Click 'Measure ROI' then click on maps.", font=("Courier", 10))
        self.lbl_res_val.pack(side=tk.RIGHT, padx=10)
        
        # 2. Canvas
        self.fig_res, self.axes_res = plt.subplots(2, 2, figsize=(10, 10))
        self.canvas_res = FigureCanvasTkAgg(self.fig_res, master=self.tab_results)
        self.canvas_res.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Bind Click on Results
        self.canvas_res.mpl_connect('button_press_event', self.on_res_click)
        
    # --- Logic ---
    
    def on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        if not paths: return
        self.listbox.delete(0, tk.END)
        self.loaded_series = []
        for p in paths:
            if os.path.isdir(p):
                print(f"Scanning: {p}")
                self.loaded_series.extend(scan_directory(p))
        for s in self.loaded_series:
            self.listbox.insert(tk.END, f"{s.modality}: {s.description} ({len(s.file_paths)} imgs)")

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        self.listbox.delete(0, tk.END)
        self.loaded_series = scan_directory(folder_path)
        for s in self.loaded_series:
            self.listbox.insert(tk.END, f"{s.modality}: {s.description} ({len(s.file_paths)} imgs)")
            
    def on_series_select(self, event):
        if self.is_loading: return
        
        selection = self.listbox.curselection()
        if not selection: return
        
        self.current_series = self.loaded_series[selection[0]]
        print(f"Loading {self.current_series.series_uid}...")
        
        # Start background thread
        self.is_loading = True
        self.progress_bar.start(10) # Indeterminate mode
        self.root.config(cursor="watch")
        
        thread = threading.Thread(target=self.load_series_thread)
        thread.daemon = True
        thread.start()
        
    def load_series_thread(self):
        try:
            data = self.current_series.load_data()
            # Schedule update on main thread
            self.root.after(0, lambda: self.on_series_loaded(data))
        except Exception as e:
            self.root.after(0, lambda: self.on_load_error(e))
            
    def on_series_loaded(self, data):
        self.pixel_data = data
        self.is_loading = False
        self.progress_bar.stop()
        self.root.config(cursor="")
        
        if self.pixel_data is None:
            messagebox.showerror("Error", "Failed to load data.")
            return
            
        # Reset View
        if self.pixel_data.ndim == 4:
            self.max_slice = self.pixel_data.shape[2] - 1
        else:
            self.max_slice = 0
            
        self.slice_slider.config(to=self.max_slice)
        self.current_slice_idx = self.max_slice // 2
        self.slice_slider.set(self.current_slice_idx)
        
        # Auto Windowing
        flat = self.pixel_data.flatten()
        p5, p95 = np.percentile(flat, 5), np.percentile(flat, 95)
        width = p95 - p5
        level = (p95 + p5) / 2
        self.wl_slider.set(level)
        self.ww_slider.set(width)
        
        self.reset_view()
        self.update_viewer()
        
    def on_load_error(self, error):
        self.is_loading = False
        self.progress_bar.stop()
        self.root.config(cursor="")
        messagebox.showerror("Error", f"Failed to load series: {error}")

    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_viewer()
        
    def on_slice_change(self, val):
        self.current_slice_idx = int(val)
        self.update_viewer()
        
    def on_window_change(self, val):
        self.window_level = self.wl_slider.get()
        self.window_width = self.ww_slider.get()
        self.update_viewer()
        
    def zoom(self, factor):
        self.zoom_level *= factor
        if self.zoom_level < 0.1: self.zoom_level = 0.1
        if self.zoom_level > 20: self.zoom_level = 20
        self.update_viewer()

    def on_scroll(self, event):
        if event.inaxes != self.ax_viewer: return
        
        # Check for Control key (Zoom)
        if event.key == 'control':
            # Zoom
            base_scale = 1.1
            if event.button == 'up':
                scale_factor = 1 / base_scale
            else:
                scale_factor = base_scale
                
            self.zoom_level *= scale_factor
            
            # Limit zoom
            if self.zoom_level < 0.1: self.zoom_level = 0.1
            if self.zoom_level > 20: self.zoom_level = 20
            
            # TODO: Zoom towards mouse pointer
            # For now, zoom center
            self.update_viewer()
            
        else:
            # Slice Change
            if event.button == 'up':
                delta = 1
            else:
                delta = -1
                
            new_idx = self.current_slice_idx + delta
            if 0 <= new_idx <= self.max_slice:
                self.slice_slider.set(new_idx)
                
    def on_mouse_press(self, event):
        if event.inaxes != self.ax_viewer: return
        
        # Left Click: 
        # 1. If Shift is held -> ROI
        # 2. Else -> Start Pan
        
        if event.button == 1: # Left Click
            if event.key == 'shift':
                # ROI Logic handled in on_click, but we need to pass it or set mode
                # Actually, ROI selection is mode-based (button click sets mode)
                # Let's keep ROI selection explicit via buttons for precision.
                # But user asked for "click to drag".
                # So: Normal Left Drag = Pan.
                # ROI Selection: Click button first, then Click on image.
                
                if self.current_roi_type:
                    # ROI mode active, handled by mpl_connect in start_roi_selection?
                    # start_roi_selection connects 'button_press_event' to on_click
                    # We are overriding it here?
                    # No, mpl supports multiple callbacks.
                    # BUT we should block pan if in ROI mode?
                    pass
                else:
                    # Pan Start
                    self.last_mouse_x = event.x
                    self.last_mouse_y = event.y
                    self.root.config(cursor="fleur")
            else:
                # Normal Left Click -> Pan (if not ROI mode)
                if not self.current_roi_type:
                    self.last_mouse_x = event.x
                    self.last_mouse_y = event.y
                    self.root.config(cursor="fleur")
        
        elif event.button == 3: # Right Click
             # Optional: Context menu or alternative pan
             pass

    def on_mouse_release(self, event):
        self.last_mouse_x = None
        self.last_mouse_y = None
        if not self.current_roi_type:
            self.root.config(cursor="")

    def on_mouse_drag(self, event):
        if event.inaxes != self.ax_viewer: return
        
        # Left Drag -> Pan
        if event.button == 1: 
            if self.last_mouse_x is None: return
            if self.current_roi_type: return # Don't pan while placing ROI
            
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            
            # Convert screen pixels to data units
            xlim = self.ax_viewer.get_xlim()
            ylim = self.ax_viewer.get_ylim()
            bbox = self.ax_viewer.get_window_extent()
            
            scale_x = abs(xlim[1] - xlim[0]) / bbox.width
            scale_y = abs(ylim[1] - ylim[0]) / bbox.height
            
            self.pan_x -= dx * scale_x
            self.pan_y += dy * scale_y 
            
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            
            self.update_viewer()
            
    # Removed old on_mouse_wheel logic as replaced by on_scroll
    
    def get_current_slice_data(self):
        if self.pixel_data is None: return None
        if self.pixel_data.ndim == 4:
            # (x, y, z, t)
            # Time Average for visualization
            return np.mean(self.pixel_data[:, :, self.current_slice_idx, :], axis=-1)
        elif self.pixel_data.ndim == 3:
            return np.mean(self.pixel_data, axis=-1)
        return None
        
    def update_viewer(self):
        img = self.get_current_slice_data()
        if img is None: return
        
        # Apply Windowing
        wl = self.window_level
        ww = max(1, self.window_width)
        vmin = wl - ww / 2
        vmax = wl + ww / 2
        
        self.ax_viewer.clear()
        self.ax_viewer.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        
        # Apply Pan/Zoom
        rows, cols = img.shape
        center_x = cols / 2
        center_y = rows / 2
        
        # Effective width/height
        eff_w = cols * self.zoom_level
        eff_h = rows * self.zoom_level
        
        # Limits
        # x_min = center_x - eff_w / 2 + self.pan_x
        # x_max = center_x + eff_w / 2 + self.pan_x
        # In Matplotlib imshow, x is 0..cols, y is 0..rows (usually inverted)
        
        # Let's define zoom as magnification.
        # zoom=1 -> full image.
        # zoom=2 -> show half image.
        # View width = cols / zoom_level
        
        view_w = cols / self.zoom_level
        view_h = rows / self.zoom_level
        
        x0 = center_x - view_w / 2 + self.pan_x
        x1 = center_x + view_w / 2 + self.pan_x
        y0 = center_y - view_h / 2 + self.pan_y
        y1 = center_y + view_h / 2 + self.pan_y
        
        self.ax_viewer.set_xlim(x0, x1)
        self.ax_viewer.set_ylim(y1, y0) # Inverted Y for images
        
        self.ax_viewer.axis('off')
        self.ax_viewer.set_title(f"Slice {self.current_slice_idx+1}/{self.max_slice+1} [WL:{int(wl)} WW:{int(ww)}] (Zoom: {self.zoom_level:.1f}x)")
        
        if self.aif_coords:
            self.ax_viewer.plot(self.aif_coords[1], self.aif_coords[0], 'r+', markersize=10, label='AIF')
        if self.vof_coords:
            self.ax_viewer.plot(self.vof_coords[1], self.vof_coords[0], 'b+', markersize=10, label='VOF')
            
        if self.aif_coords or self.vof_coords:
            self.ax_viewer.legend()
            
        self.canvas_viewer.draw()
        
    def start_roi_selection(self, roi_type):
        if self.pixel_data is None: return
        self.current_roi_type = roi_type
        self.roi_cid = self.canvas_viewer.mpl_connect('button_press_event', self.on_click)
        self.root.config(cursor="cross")
        
    def on_click(self, event):
        if event.inaxes != self.ax_viewer: return
        if event.button != 1: return # Only Left Click for ROI
        
        # If in ROI mode (set by buttons), Shift key not needed.
        # Just click.
        
        # But wait, if we are in ROI mode, on_mouse_press also fires.
        # We need to ensure we don't Pan.
        # Added check in on_mouse_press/drag: if self.current_roi_type, don't pan.
        
        x, y = int(event.xdata), int(event.ydata)
        
        if self.current_roi_type == 'aif':
            self.aif_coords = (y, x)
        elif self.current_roi_type == 'vof':
            self.vof_coords = (y, x)
            
        self.canvas_viewer.mpl_disconnect(self.roi_cid)
        self.roi_cid = None
        self.current_roi_type = None # Reset ROI mode
        self.root.config(cursor="")
        self.update_viewer()
        
        # Auto switch to TDC Tab and Plot
        self.plot_tdc()
        self.notebook.select(self.tab_tdc)
        
    def plot_tdc(self):
        self.ax_tdc.clear()
        if self.pixel_data is None: return
        
        # Extract data from CURRENT SLICE for now
        # If AIF is selected on slice Z, we use that Z.
        # Ideally AIF should be consistent, but let's assume slice-specific or user stays on same slice.
        # Better: Store slice idx of ROI? 
        # For now, extract from the slice currently displayed or where ROI was clicked.
        # Assuming user doesn't change slice between click and plot (immediate).
        
        if self.pixel_data.ndim == 4:
            vol = self.pixel_data[:, :, self.current_slice_idx, :]
        else:
            vol = self.pixel_data
            
        t = np.arange(vol.shape[-1])
        
        if self.aif_coords:
            y = vol[self.aif_coords[0], self.aif_coords[1], :]
            self.ax_tdc.plot(t, y, 'r-o', label='AIF', linewidth=2)
            
        if self.vof_coords:
            y = vol[self.vof_coords[0], self.vof_coords[1], :]
            self.ax_tdc.plot(t, y, 'b-o', label='VOF', linewidth=2)
            
        self.ax_tdc.set_title(f"Time-Density Curve (Slice {self.current_slice_idx})")
        self.ax_tdc.set_xlabel("Time Frame")
        self.ax_tdc.set_ylabel("Intensity")
        self.ax_tdc.grid(True)
        self.ax_tdc.legend()
        self.canvas_tdc.draw()
        
    def calculate(self):
        if self.pixel_data is None or self.aif_coords is None:
            messagebox.showwarning("Warning", "Select AIF first.")
            return
            
        modality = self.modality_var.get()
        dt = self.current_series.metadata.get('dt', 1.0)
        
        try:
            # Process Current Slice
            if self.pixel_data.ndim == 4:
                calc_data = self.pixel_data[:, :, self.current_slice_idx, :]
            else:
                calc_data = self.pixel_data
                
            # Mask (simple)
            mean_img = np.mean(calc_data, axis=-1)
            mask = mean_img > (0.1 * np.max(mean_img))
            
            print("Calculating Perfusion...")
            if modality == "CT":
                res = calculate_ct_perfusion(calc_data, self.aif_coords, mask=mask, dt=dt)
            else:
                te = self.current_series.metadata.get('TE', 0.030)
                res = calculate_mri_perfusion(calc_data, self.aif_coords, te=te, mask=mask, dt=dt)
                
            self.show_results_in_tab(res)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()
            
    def show_results_in_tab(self, res):
        # Plot to Tab 3
        axes = self.axes_res.flatten()
        for ax in axes: ax.clear()
        
        maps = [('CBF', res['cbf']), ('CBV', res['cbv']), 
                ('MTT', res['mtt']), ('Tmax/TTP', res.get('tmax', res.get('ttp')))]
        
        for i, (name, data) in enumerate(maps):
            im = axes[i].imshow(data, cmap='jet')
            axes[i].set_title(name)
            axes[i].axis('off')
            
            # Apply View Transform (Sync with Viewer)
            rows, cols = data.shape
            center_x = cols / 2
            center_y = rows / 2
            
            view_w = cols / self.zoom_level
            view_h = rows / self.zoom_level
            
            x0 = center_x - view_w / 2 + self.pan_x
            x1 = center_x + view_w / 2 + self.pan_x
            y0 = center_y - view_h / 2 + self.pan_y
            y1 = center_y + view_h / 2 + self.pan_y
            
            axes[i].set_xlim(x0, x1)
            axes[i].set_ylim(y1, y0)
            
        self.canvas_res.draw()
        self.notebook.select(self.tab_results) # Auto switch
        
        # Store results for interaction
        self.current_results = res
        
    def toggle_measure_mode(self):
        self.is_measuring = not self.is_measuring
        if self.is_measuring:
            self.btn_measure.config(text="Stop Measuring", relief=tk.SUNKEN)
            self.root.config(cursor="cross")
            self.lbl_res_val.config(text="Mode: Measuring. Click on map.")
        else:
            self.btn_measure.config(text="Measure ROI", relief=tk.RAISED)
            self.root.config(cursor="")
            self.lbl_res_val.config(text="")
            
    def clear_measure_roi(self):
        # Remove patches
        for p in self.res_roi_patches:
            p.remove()
        self.res_roi_patches = []
        self.canvas_res.draw()
        self.lbl_res_val.config(text="")
        
    def on_res_click(self, event):
        if event.inaxes not in self.axes_res.flatten(): return
        if event.button != 1: return
        if not self.is_measuring: return
        
        x, y = int(event.xdata), int(event.ydata)
        
        # Draw ROI (Circle) on ALL axes to show sync
        # First clear old ones? Or allow multiple? 
        # User said "hook drawing", usually implies one active measurement or multiple.
        # Let's clear previous for simplicity unless we track them.
        self.clear_measure_roi() 
        
        import matplotlib.patches as patches
        
        # Measure 3x3 Mean
        names = ['CBF', 'CBV', 'MTT', 'Tmax']
        
        if not hasattr(self, 'current_results'): return
        
        res = self.current_results
        maps = [res['cbf'], res['cbv'], res['mtt'], res.get('tmax', res.get('ttp'))]
        
        res_str = f"Pos ({x},{y}): "
        
        for i, (name, data) in enumerate(zip(names, maps)):
            map_name, map_data = name, data[0] # zip items are (name, data)
            
            # Correct unpacking
            current_name = name
            current_data = data
            
            # Draw Circle
            ax = self.axes_res.flatten()[i]
            circ = patches.Circle((x, y), radius=2, edgecolor='red', facecolor='none', linewidth=1.5)
            ax.add_patch(circ)
            self.res_roi_patches.append(circ)
            
            # Calculate
            h, w = current_data.shape
            y1, y2 = max(0, y-1), min(h, y+2)
            x1, x2 = max(0, x-1), min(w, x+2)
            
            patch = current_data[y1:y2, x1:x2]
            mean_val = np.mean(patch)
            res_str += f"{current_name}={mean_val:.2f}  "
            
        self.canvas_res.draw()
        self.lbl_res_val.config(text=res_str)
        
        # Turn off measure mode after one click? Or keep it?
        # Keep it to allow measuring another point easily.
        # self.toggle_measure_mode() 

if __name__ == "__main__":
    print("Starting GUI...")
    try:
        root = TkinterDnD.Tk()
        app = PerfusionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
