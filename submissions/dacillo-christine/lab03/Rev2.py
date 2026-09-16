import sys
import os
import numpy as np
import pandas as pd

# PySide6 Qt GUI Imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QDockWidget, QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton,
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox, QToolBar,
    QFileDialog, QMessageBox, QHeaderView, QSplitter, QFrame
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QFont, QAction, QColor

# Matplotlib Qt Integration
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D


# ==============================================================================
# 1. STRUCTURAL ANALYSIS ENGINE (DIRECT STIFFNESS METHOD)
# ==============================================================================
class DirectStiffness3DSolver:
    """
    REV 2 - 3D Space Frame Direct Stiffness Solver.
    Global Coordinate System: X (lateral), Y (vertical), Z (lateral depth).
    6 DOFs per node: [UX, UY, UZ, RX, RY, RZ]
    """
    def __init__(self, nodes, members, supports, loads, materials, sections, unit_system="Standard Metric"):
        self.nodes = nodes          # Dict: {node_id: (x, y, z)}
        self.members = members      # List of dicts
        self.supports = supports    # Dict: {node_id: [ux, uy, uz, rx, ry, rz]} (1=fixed, 0=free)
        self.loads = loads          # Dict: {node_id: [fx, fy, fz, mx, my, mz]}
        self.materials = materials  # Dict of material properties
        self.sections = sections    # Dict of section properties
        self.unit_system = unit_system

        self.num_nodes = len(self.nodes)
        self.num_dof = self.num_nodes * 6
        self.K_global = None
        self.F_global = None
        self.displacements = None
        self.reactions = None

    def get_element_transformation(self, ni_coords, nj_coords, beta_deg=0.0):
        """Calculates 12x12 transformation matrix T incorporating Beta angle."""
        xi, yi, zi = ni_coords
        xj, yj, zj = nj_coords
        dx, dy, dz = xj - xi, yj - yi, zj - zi
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        if L == 0:
            raise ValueError("Member length cannot be zero.")

        cx, cy, cz = dx / L, dy / L, dz / L
        beta = np.radians(beta_deg)

        # Right-handed orientation matrix r0
        if np.isclose(cx, 0.0) and np.isclose(cz, 0.0):
            if cy > 0:
                r0 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            else:
                r0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        else:
            D = np.sqrt(cx**2 + cz**2)
            r0 = np.array([
                [cx, cy, cz],
                [-cx * cy / D, D, -cy * cz / D],
                [-cz / D, 0, cx / D]
            ])

        # Apply Beta angle rotation around member local x-axis
        R_beta = np.array([
            [1, 0, 0],
            [0, np.cos(beta), np.sin(beta)],
            [0, -np.sin(beta), np.cos(beta)]
        ])
        r = R_beta @ r0

        T = np.zeros((12, 12))
        for i in range(4):
            T[i*3:(i+1)*3, i*3:(i+1)*3] = r
        return T, L

    def assemble_and_solve(self):
        """Assembles global stiffness matrix [K], load vector {F}, and solves [K]{D} = {F}."""
        self.K_global = np.zeros((self.num_dof, self.num_dof))
        self.F_global = np.zeros(self.num_dof)

        node_id_map = {nid: idx for idx, nid in enumerate(sorted(self.nodes.keys()))}

        # Load Vector Assembly
        for nid, f_vec in self.loads.items():
            if nid in node_id_map:
                n_idx = node_id_map[nid]
                self.F_global[n_idx*6 : (n_idx+1)*6] = f_vec

        # Stiffness Assembly
        for mem in self.members:
            ni, nj = mem['ni'], mem['nj']
            idx_i = node_id_map[ni]
            idx_j = node_id_map[nj]
            dof_indices = list(range(idx_i*6, (idx_i+1)*6)) + list(range(idx_j*6, (idx_j+1)*6))

            mat = self.materials.get(mem['mat'], {'E': 200e9, 'G': 79.3e9})
            sec = self.sections.get(mem['sec'], {'A': 0.00171, 'Iy': 1.66e-6, 'Iz': 5.58e-6, 'J': 2.84e-8})

            E, G = mat['E'], mat['G']
            A, Iy, Iz, J = sec['A'], sec['Iy'], sec['Iz'], sec['J']

            T, L = self.get_element_transformation(self.nodes[ni], self.nodes[nj], mem.get('beta', 0.0))

            # Local 12x12 Stiffness Matrix
            k_loc = np.zeros((12, 12))
            
            # Axial
            EA_L = E * A / L
            k_loc[0, 0] = k_loc[6, 6] = EA_L
            k_loc[0, 6] = k_loc[6, 0] = -EA_L
            
            # Torsion
            GJ_L = G * J / L
            k_loc[3, 3] = k_loc[9, 9] = GJ_L
            k_loc[3, 9] = k_loc[9, 3] = -GJ_L
            
            # Bending about local z (uy, rz)
            EIz = E * Iz
            k_loc[1, 1] = k_loc[7, 7] = 12 * EIz / L**3
            k_loc[1, 7] = k_loc[7, 1] = -12 * EIz / L**3
            k_loc[1, 5] = k_loc[5, 1] = k_loc[1, 11] = k_loc[11, 1] = 6 * EIz / L**2
            k_loc[7, 5] = k_loc[5, 7] = -6 * EIz / L**2
            k_loc[7, 11] = k_loc[11, 7] = -6 * EIz / L**2
            k_loc[5, 5] = k_loc[11, 11] = 4 * EIz / L
            k_loc[5, 11] = k_loc[11, 5] = 2 * EIz / L
            
            # Bending about local y (uz, ry)
            EIy = E * Iy
            k_loc[2, 2] = k_loc[8, 8] = 12 * EIy / L**3
            k_loc[2, 8] = k_loc[8, 2] = -12 * EIy / L**3
            k_loc[2, 4] = k_loc[4, 2] = -6 * EIy / L**2
            k_loc[2, 10] = k_loc[10, 2] = -6 * EIy / L**2
            k_loc[8, 4] = k_loc[4, 8] = 6 * EIy / L**2
            k_loc[8, 10] = k_loc[10, 8] = 6 * EIy / L**2
            k_loc[4, 4] = k_loc[10, 10] = 4 * EIy / L
            k_loc[4, 10] = k_loc[10, 4] = 2 * EIy / L

            # Global stiffness contribution
            k_glob = T.T @ k_loc @ T

            for r_i, gi in enumerate(dof_indices):
                for c_i, gj in enumerate(dof_indices):
                    self.K_global[gi, gj] += k_glob[r_i, c_i]

        # Apply Boundary Conditions (Penalty Approach)
        penalty = 1e15
        K_mod = self.K_global.copy()
        F_mod = self.F_global.copy()

        for nid, supp_vec in self.supports.items():
            if nid in node_id_map:
                n_idx = node_id_map[nid]
                for dof_i, constrained in enumerate(supp_vec):
                    if constrained:
                        idx = n_idx * 6 + dof_i
                        K_mod[idx, idx] += penalty
                        F_mod[idx] = 0.0

        # Solve Linear System
        try:
            self.displacements = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            raise ValueError("Stiffness matrix is singular. Ensure structure has proper boundary supports.")

        self.reactions = self.K_global @ self.displacements - self.F_global
        return self.displacements, self.reactions


# ==============================================================================
# 2. EMBEDDED MATPLOTLIB 3D CANVAS (ESTHETIC DEEP PLUM & ROSE PALETTE)
# ==============================================================================
class Structure3DCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = plt.figure(figsize=(8, 6), facecolor='#fcf4f6')
        self.ax = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)

    def plot_structure(self, nodes, members, displacements=None, scale=50.0,
                       show_node_labels=True, show_mem_labels=True,
                       show_loads=True, show_deformed=True, loads=None, supports=None):
        self.ax.clear()
        
        # Deep Plum background for high contrast with soft pinks, oranges & yellows
        self.ax.set_facecolor('#200b21')  
        self.ax.grid(True, color='#4a2c52', linestyle=':', alpha=0.7)

        node_id_map = {nid: idx for idx, nid in enumerate(sorted(nodes.keys()))}

        # Plot Undeformed Structure (Light Rose/Pink: #f78fb3)
        for mem in members:
            ni, nj = mem['ni'], mem['nj']
            xi, yi, zi = nodes[ni]
            xj, yj, zj = nodes[nj]

            self.ax.plot([xi, xj], [zi, zj], [yi, yj], color='#f78fb3', linewidth=2.5, marker='o', markersize=4)

            if show_mem_labels:
                mx, my, mz = (xi + xj)/2, (yi + yj)/2, (zi + zj)/2
                self.ax.text(mx, mz, my, f"M{mem['id']}", color='#f39c12', fontsize=8, weight='bold')

        # Plot Deformed Structure (Vibrant Sunset Orange/Coral: #ff7e5f)
        if show_deformed and displacements is not None:
            for mem in members:
                ni, nj = mem['ni'], mem['nj']
                idx_i, idx_j = node_id_map[ni], node_id_map[nj]

                ux_i, uy_i, uz_i = displacements[idx_i*6 : idx_i*6+3] * scale
                ux_j, uy_j, uz_j = displacements[idx_j*6 : idx_j*6+3] * scale

                xi_def, yi_def, zi_def = nodes[ni][0] + ux_i, nodes[ni][1] + uy_i, nodes[ni][2] + uz_i
                xj_def, yj_def, zj_def = nodes[nj][0] + ux_j, nodes[nj][1] + uy_j, nodes[nj][2] + uz_j

                self.ax.plot([xi_def, xj_def], [zi_def, zj_def], [yi_def, yj_def],
                             color='#ff7e5f', linestyle='--', linewidth=2.2)

        # Plot Nodes and Supports
        for nid, (x, y, z) in nodes.items():
            is_supported = supports and nid in supports and any(supports[nid])
            marker_color = '#d88dfa' if is_supported else '#ffffff'  # Light purple for supports, white for nodes
            self.ax.scatter([x], [z], [y], color=marker_color, s=40 if is_supported else 25)

            if show_node_labels:
                self.ax.text(x, z, y, f"  N{nid}", color='#ffffff', fontsize=8, weight='bold')

        # Plot Loads (Vibrant Gold/Yellow Quiver Arrows: #ffd166)
        if show_loads and loads:
            for nid, f_vec in loads.items():
                if any(f_vec[:3]):
                    x, y, z = nodes[nid]
                    fx, fy, fz = f_vec[:3]
                    mag = np.sqrt(fx**2 + fy**2 + fz**2)
                    if mag > 0:
                        dfx, dfy, dfz = (fx/mag)*1.2, (fy/mag)*1.2, (fz/mag)*1.2
                        self.ax.quiver(x, z, y, dfx, dfz, dfy, color='#ffd166', length=1.0, normalize=True)

        # Axis Styling
        self.ax.set_xlabel("X (m)", color='#f78fb3', fontsize=9, weight='bold')
        self.ax.set_ylabel("Z (m)", color='#f78fb3', fontsize=9, weight='bold')
        self.ax.set_zlabel("Y [vertical] (m)", color='#f78fb3', fontsize=9, weight='bold')
        self.ax.tick_params(colors='#e8b0bd', labelsize=8)

        self.ax.set_title("REV 2 — 3D Structural Frame Visualization", color='#6c2350', fontsize=11, weight='bold', pad=10)
        self.draw()


# ==============================================================================
# 3. PYSIDE6 MAIN APPLICATION WINDOW (PROFESSIONAL PINK-PURPLE THEME)
# ==============================================================================
class Rev2SolverWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("REV 2 — 3D Frame Matrix Solver")
        self.resize(1400, 850)

        # Default Benchmark Dimensions (6m x 6m x 6m Cube)
        self.dim_x = 6.0
        self.dim_y = 6.0
        self.dim_z = 6.0
        self.unit_system = "Standard Metric"
        self.selected_material = "ASTM A36 Steel"
        self.selected_section = "W150X13.5"
        self.def_scale = 50.0

        # Databases
        self.materials_db = {
            "ASTM A36 Steel": {'E': 200e9, 'G': 79.3e9, 'Fy': 250e6, 'Poisson': 0.26},
            "Grade 50 Steel": {'E': 200e9, 'G': 79.3e9, 'Fy': 345e6, 'Poisson': 0.26}
        }
        self.sections_db = {
            "W150X13.5": {'A': 0.00171, 'Iy': 1.66e-6, 'Iz': 5.58e-6, 'J': 2.84e-8, 'd': 150, 'bf': 100},
            "W200X15": {'A': 0.00191, 'Iy': 2.12e-6, 'Iz': 8.51e-6, 'J': 3.50e-8, 'd': 200, 'bf': 100}
        }

        self.solver = None
        self.displacements = None
        self.reactions = None

        self.init_default_benchmark()
        self.setup_ui()
        self.update_3d_plot()

    def init_default_benchmark(self):
        """Initializes the validated 6m x 6m x 6m cube benchmark model."""
        x, y, z = self.dim_x, self.dim_y, self.dim_z

        # 8 Nodes
        self.nodes = {
            1: (0.0, 0.0, 0.0), 2: (x, 0.0, 0.0), 3: (x, 0.0, z), 4: (0.0, 0.0, z),
            5: (0.0, y, 0.0), 6: (x, y, 0.0), 7: (x, y, z), 8: (0.0, y, z)
        }

        # 12 Members
        self.members = [
            {'id': 1, 'ni': 1, 'nj': 2, 'type': 'Beam', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 2, 'ni': 2, 'nj': 3, 'type': 'Beam', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 3, 'ni': 3, 'nj': 4, 'type': 'Beam', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 4, 'ni': 4, 'nj': 1, 'type': 'Beam', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 5, 'ni': 5, 'nj': 6, 'type': 'Beam', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 6, 'ni': 6, 'nj': 7, 'type': 'Beam', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 7, 'ni': 7, 'nj': 8, 'type': 'Beam', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 8, 'ni': 8, 'nj': 5, 'type': 'Beam', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 9, 'ni': 1, 'nj': 5, 'type': 'Column', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 10, 'ni': 2, 'nj': 6, 'type': 'Column', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 11, 'ni': 3, 'nj': 7, 'type': 'Column', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0},
            {'id': 12, 'ni': 4, 'nj': 8, 'type': 'Column', 'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0}
        ]

        # Pinned Base Supports (Nodes 1-4)
        self.supports = {
            1: [1, 1, 1, 0, 0, 0], 2: [1, 1, 1, 0, 0, 0],
            3: [1, 1, 1, 0, 0, 0], 4: [1, 1, 1, 0, 0, 0]
        }

        # Lateral Nodal Load at Node 5
        self.loads = {5: [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

    def setup_ui(self):
        # Professional White-Pink-Purple Theme with Orange/Yellow Highlights
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #fcf4f6; 
            }
            QToolBar { 
                background-color: #6c2350; 
                border: none; 
                padding: 6px; 
            }
            QToolBar QPushButton { 
                color: #ffffff; 
                background-color: #8c265c; 
                border: 1px solid #a83672;
                border-radius: 4px; 
                padding: 6px 12px; 
                font-weight: bold; 
            }
            QToolBar QPushButton:hover { 
                background-color: #ff7e5f; 
                border-color: #ffd166;
                color: #ffffff; 
            }
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #e8b0bd; 
                border-radius: 8px; 
                margin-top: 10px; 
                padding-top: 12px;
                background-color: #ffffff; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 12px; 
                color: #6c2350; 
            }
            QPushButton { 
                background-color: #8c265c; 
                color: #ffffff; 
                border-radius: 5px; 
                padding: 7px; 
                font-weight: bold; 
            }
            QPushButton:hover { 
                background-color: #ff7e5f; 
                color: #ffffff; 
            }
            QTableWidget { 
                gridline-color: #f2d6dc; 
                background-color: #ffffff;
                selection-background-color: #f78fb3;
                selection-color: #200b21;
                border: 1px solid #e8b0bd;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #6c2350;
                color: #ffffff;
                padding: 5px;
                font-weight: bold;
                border: none;
            }
            QTabWidget::pane {
                border: 1px solid #e8b0bd;
                border-radius: 6px;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background: #f4dbe1;
                color: #6c2350;
                padding: 8px 16px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #8c265c;
                color: #ffffff;
            }
            QComboBox, QDoubleSpinBox {
                background-color: #ffffff;
                border: 1px solid #e8b0bd;
                border-radius: 4px;
                padding: 4px;
                color: #333333;
            }
            QComboBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #8c265c;
            }
        """)

        # Toolbar
        toolbar = QToolBar("REV 2 Action Bar", self)
        self.addToolBar(toolbar)

        btn_run = QPushButton("Run Analysis")
        btn_run.clicked.connect(self.run_analysis)
        toolbar.addWidget(btn_run)

        toolbar.addSeparator()
        btn_reset = QPushButton("Reset Benchmark (6m Cube)")
        btn_reset.clicked.connect(self.reset_benchmark)
        toolbar.addWidget(btn_reset)

        toolbar.addSeparator()
        btn_save = QPushButton("Save Input (.xlsx)")
        btn_save.clicked.connect(self.save_input_excel)
        toolbar.addWidget(btn_save)

        btn_export = QPushButton("Export Results (.xlsx)")
        btn_export.clicked.connect(self.export_results_excel)
        toolbar.addWidget(btn_export)

        # Splitter Layout
        splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(splitter)

        # Left Controls Dock
        left_dock = QWidget()
        left_layout = QVBoxLayout(left_dock)

        # Geometry Box
        geom_box = QGroupBox("Model Geometry")
        geom_grid = QGridLayout(geom_box)

        geom_grid.addWidget(QLabel("X Span (m):"), 0, 0)
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(1.0, 100.0)
        self.spin_x.setValue(self.dim_x)
        self.spin_x.valueChanged.connect(self.update_geometry_from_inputs)
        geom_grid.addWidget(self.spin_x, 0, 1)

        geom_grid.addWidget(QLabel("Y Height (m):"), 1, 0)
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(1.0, 100.0)
        self.spin_y.setValue(self.dim_y)
        self.spin_y.valueChanged.connect(self.update_geometry_from_inputs)
        geom_grid.addWidget(self.spin_y, 1, 1)

        geom_grid.addWidget(QLabel("Z Depth (m):"), 2, 0)
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setRange(1.0, 100.0)
        self.spin_z.setValue(self.dim_z)
        self.spin_z.valueChanged.connect(self.update_geometry_from_inputs)
        geom_grid.addWidget(self.spin_z, 2, 1)

        left_layout.addWidget(geom_box)

        # Section & Material Database Box
        db_box = QGroupBox("Section & Material Libraries")
        db_grid = QGridLayout(db_box)

        db_grid.addWidget(QLabel("Unit System:"), 0, 0)
        self.combo_units = QComboBox()
        self.combo_units.addItems(["Standard Metric", "Imperial"])
        db_grid.addWidget(self.combo_units, 0, 1)

        db_grid.addWidget(QLabel("Material:"), 1, 0)
        self.combo_mat = QComboBox()
        self.combo_mat.addItems(list(self.materials_db.keys()))
        db_grid.addWidget(self.combo_mat, 1, 1)

        db_grid.addWidget(QLabel("Section Profile:"), 2, 0)
        self.combo_sec = QComboBox()
        self.combo_sec.addItems(list(self.sections_db.keys()))
        db_grid.addWidget(self.combo_sec, 2, 1)

        left_layout.addWidget(db_box)

        # Visual Controls Box
        disp_box = QGroupBox("3D View Controls & Scale")
        disp_grid = QGridLayout(disp_box)

        self.chk_nodes = QCheckBox("Show Node Labels")
        self.chk_nodes.setChecked(True)
        self.chk_nodes.stateChanged.connect(self.update_3d_plot)
        disp_grid.addWidget(self.chk_nodes, 0, 0)

        self.chk_mems = QCheckBox("Show Member Labels")
        self.chk_mems.setChecked(True)
        self.chk_mems.stateChanged.connect(self.update_3d_plot)
        disp_grid.addWidget(self.chk_mems, 0, 1)

        self.chk_deformed = QCheckBox("Show Deformed Shape")
        self.chk_deformed.setChecked(True)
        self.chk_deformed.stateChanged.connect(self.update_3d_plot)
        disp_grid.addWidget(self.chk_deformed, 1, 0)

        disp_grid.addWidget(QLabel("Deform Scale:"), 2, 0)
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(1.0, 1000.0)
        self.spin_scale.setValue(self.def_scale)
        self.spin_scale.valueChanged.connect(self.on_scale_change)
        disp_grid.addWidget(self.spin_scale, 2, 1)

        left_layout.addWidget(disp_box)
        left_layout.addStretch()
        splitter.addWidget(left_dock)

        # Center Canvas Box
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        self.canvas = Structure3DCanvas(self)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self)

        self.lbl_info = QLabel("REV 2 Ready — Default 6m x 6m x 6m Cube Benchmark Loaded.")
        self.lbl_info.setStyleSheet(
            "font-weight: bold; color: #6c2350; padding: 6px; background-color: #f4dbe1; border: 1px solid #e8b0bd; border-radius: 4px;"
        )

        center_layout.addWidget(self.canvas_toolbar)
        center_layout.addWidget(self.canvas)
        center_layout.addWidget(self.lbl_info)
        splitter.addWidget(center_widget)

        # Right Data Explorer Panel
        right_dock = QWidget()
        right_layout = QVBoxLayout(right_dock)
        self.tabs = QTabWidget()

        self.table_nodes = QTableWidget(len(self.nodes), 4)
        self.table_nodes.setHorizontalHeaderLabels(["Node ID", "X (m)", "Y (m)", "Z (m)"])
        self.populate_nodes_table()
        self.tabs.addTab(self.table_nodes, "Nodes")

        self.table_disp = QTableWidget(0, 7)
        self.table_disp.setHorizontalHeaderLabels(["Node", "UX (mm)", "UY (mm)", "UZ (mm)", "RX (rad)", "RY (rad)", "RZ (rad)"])
        self.tabs.addTab(self.table_disp, "Displacements")

        self.table_react = QTableWidget(0, 7)
        self.table_react.setHorizontalHeaderLabels(["Node", "FX (kN)", "FY (kN)", "FZ (kN)", "MX (kNm)", "MY (kNm)", "MZ (kNm)"])
        self.tabs.addTab(self.table_react, "Reactions")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_dock)

        splitter.setSizes([300, 700, 400])

    def populate_nodes_table(self):
        self.table_nodes.setRowCount(len(self.nodes))
        for row, (nid, (x, y, z)) in enumerate(self.nodes.items()):
            self.table_nodes.setItem(row, 0, QTableWidgetItem(str(nid)))
            self.table_nodes.setItem(row, 1, QTableWidgetItem(f"{x:.2f}"))
            self.table_nodes.setItem(row, 2, QTableWidgetItem(f"{y:.2f}"))
            self.table_nodes.setItem(row, 3, QTableWidgetItem(f"{z:.2f}"))

    def update_geometry_from_inputs(self):
        self.dim_x = self.spin_x.value()
        self.dim_y = self.spin_y.value()
        self.dim_z = self.spin_z.value()

        x, y, z = self.dim_x, self.dim_y, self.dim_z
        self.nodes[2] = (x, 0.0, 0.0)
        self.nodes[3] = (x, 0.0, z)
        self.nodes[4] = (0.0, 0.0, z)
        self.nodes[5] = (0.0, y, 0.0)
        self.nodes[6] = (x, y, 0.0)
        self.nodes[7] = (x, y, z)
        self.nodes[8] = (0.0, y, z)

        self.populate_nodes_table()
        self.displacements = None
        self.update_3d_plot()

    def on_scale_change(self):
        self.def_scale = self.spin_scale.value()
        self.update_3d_plot()

    def update_3d_plot(self):
        self.canvas.plot_structure(
            self.nodes, self.members, self.displacements, scale=self.def_scale,
            show_node_labels=self.chk_nodes.isChecked(),
            show_mem_labels=self.chk_mems.isChecked(),
            show_deformed=self.chk_deformed.isChecked(),
            loads=self.loads, supports=self.supports
        )

    def run_analysis(self):
        try:
            self.solver = DirectStiffness3DSolver(
                self.nodes, self.members, self.supports, self.loads,
                self.materials_db, self.sections_db, self.combo_units.currentText()
            )
            self.displacements, self.reactions = self.solver.assemble_and_solve()

            node_id_map = {nid: idx for idx, nid in enumerate(sorted(self.nodes.keys()))}
            self.table_disp.setRowCount(len(self.nodes))
            max_disp = 0.0
            max_node = 1

            for row, nid in enumerate(sorted(self.nodes.keys())):
                idx = node_id_map[nid]
                d = self.displacements[idx*6 : (idx+1)*6]
                trans_mag = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
                if trans_mag > max_disp:
                    max_disp = trans_mag
                    max_node = nid

                self.table_disp.setItem(row, 0, QTableWidgetItem(f"Node {nid}"))
                self.table_disp.setItem(row, 1, QTableWidgetItem(f"{d[0]*1e3:.4f}"))
                self.table_disp.setItem(row, 2, QTableWidgetItem(f"{d[1]*1e3:.4f}"))
                self.table_disp.setItem(row, 3, QTableWidgetItem(f"{d[2]*1e3:.4f}"))
                self.table_disp.setItem(row, 4, QTableWidgetItem(f"{d[3]:.6f}"))
                self.table_disp.setItem(row, 5, QTableWidgetItem(f"{d[4]:.6f}"))
                self.table_disp.setItem(row, 6, QTableWidgetItem(f"{d[5]:.6f}"))

            self.table_react.setRowCount(len(self.supports))
            for row, nid in enumerate(sorted(self.supports.keys())):
                idx = node_id_map[nid]
                r = self.reactions[idx*6 : (idx+1)*6]
                self.table_react.setItem(row, 0, QTableWidgetItem(f"Node {nid}"))
                self.table_react.setItem(row, 1, QTableWidgetItem(f"{r[0]/1e3:.2f}"))
                self.table_react.setItem(row, 2, QTableWidgetItem(f"{r[1]/1e3:.2f}"))
                self.table_react.setItem(row, 3, QTableWidgetItem(f"{r[2]/1e3:.2f}"))
                self.table_react.setItem(row, 4, QTableWidgetItem(f"{r[3]/1e3:.2f}"))
                self.table_react.setItem(row, 5, QTableWidgetItem(f"{r[4]/1e3:.2f}"))
                self.table_react.setItem(row, 6, QTableWidgetItem(f"{r[5]/1e3:.2f}"))

            self.lbl_info.setText(
                f"REV 2 Analysis Complete! Max Translation: {max_disp*1e3:.3f} mm at Node {max_node}. "
                f"Material: {self.combo_mat.currentText()} | Section: {self.combo_sec.currentText()}"
            )
            self.update_3d_plot()
            QMessageBox.information(self, "Success", "REV 2 - 3D Frame Direct Stiffness Analysis completed successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to solve structure:\n{str(e)}")

    def reset_benchmark(self):
        self.spin_x.setValue(6.0)
        self.spin_y.setValue(6.0)
        self.spin_z.setValue(6.0)
        self.combo_units.setCurrentIndex(0)
        self.combo_mat.setCurrentIndex(0)
        self.combo_sec.setCurrentIndex(0)
        self.init_default_benchmark()
        self.displacements = None
        self.populate_nodes_table()
        self.update_3d_plot()
        self.lbl_info.setText("Reset to REV 2 Benchmark Model (6m x 6m x 6m Cube).")

    def save_input_excel(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Structural Input", "Structural_Input_Rev_2.xlsx", "Excel Files (*.xlsx)")
        if filepath:
            with pd.ExcelWriter(filepath) as writer:
                pd.DataFrame([{'System': self.combo_units.currentText(), 'Material': self.combo_mat.currentText(), 'Section': self.combo_sec.currentText()}]).to_excel(writer, sheet_name='Settings', index=False)
                pd.DataFrame([{'Node': k, 'X': v[0], 'Y': v[1], 'Z': v[2]} for k, v in self.nodes.items()]).to_excel(writer, sheet_name='Nodes', index=False)
                pd.DataFrame(self.members).to_excel(writer, sheet_name='Members', index=False)
            QMessageBox.information(self, "Saved", f"Input file saved to:\n{filepath}")

    def export_results_excel(self):
        if self.displacements is None:
            QMessageBox.warning(self, "No Results", "Please run the analysis before exporting results.")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Export Results", "Structural_Results_Rev_2.xlsx", "Excel Files (*.xlsx)")
        if filepath:
            with pd.ExcelWriter(filepath) as writer:
                disp_data = []
                node_id_map = {nid: idx for idx, nid in enumerate(sorted(self.nodes.keys()))}
                for nid in sorted(self.nodes.keys()):
                    idx = node_id_map[nid]
                    d = self.displacements[idx*6 : (idx+1)*6]
                    disp_data.append({'Node': nid, 'UX_mm': d[0]*1e3, 'UY_mm': d[1]*1e3, 'UZ_mm': d[2]*1e3, 'RX_rad': d[3], 'RY_rad': d[4], 'RZ_rad': d[5]})
                pd.DataFrame(disp_data).to_excel(writer, sheet_name='Displacements', index=False)
            QMessageBox.information(self, "Exported", f"Results exported to:\n{filepath}")


# ==============================================================================
# 4. LAUNCHER ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Rev2SolverWindow()
    window.show()
    sys.exit(app.exec())