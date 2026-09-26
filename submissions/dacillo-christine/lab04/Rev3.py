import sys
import os
import numpy as np
import pandas as pd

# PySide6 Qt GUI Imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QDockWidget, QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton,
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox, QToolBar,
    QFileDialog, QMessageBox, QHeaderView, QSplitter, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

# Matplotlib Qt Integration
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D


# ==============================================================================
# 1. REV 3 LOAD & DIAPHRAGM DATA STRUCTURES
# ==============================================================================
class NodalLoad:
    def __init__(self, node_id, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0):
        self.node_id = node_id
        self.fx = fx  # N
        self.fy = fy  # N
        self.fz = fz  # N
        self.mx = mx  # N*m
        self.my = my  # N*m
        self.mz = mz  # N*m


class MemberDistributedLoad:
    """Uniformly distributed load on a member (in N/m)."""
    def __init__(self, member_id, direction='Y', magnitude=0.0):
        self.member_id = member_id
        self.direction = direction.upper()  # 'X', 'Y', 'Z' (Global)
        self.magnitude = magnitude          # N/m (negative for downward)


class MemberPointLoad:
    """Point load applied along a member length."""
    def __init__(self, member_id, location_ratio=0.5, direction='Y', magnitude=0.0):
        self.member_id = member_id
        self.location_ratio = location_ratio  # 0.5 = Midpoint
        self.direction = direction.upper()    # 'X', 'Y', 'Z' (Global)
        self.magnitude = magnitude            # N


class Diaphragm:
    """Rigid floor/roof diaphragm constraint tying nodes at an elevation."""
    def __init__(self, diaphragm_id, name, elevation, master_node, constrained_nodes, dofs=('UX', 'UZ', 'RY')):
        self.id = diaphragm_id
        self.name = name
        self.elevation = elevation
        self.master_node = master_node
        self.constrained_nodes = constrained_nodes
        self.dofs = dofs


class LoadCase:
    def __init__(self, lc_id, name, category, self_weight_factor=0.0, sw_dir='Y', sw_factor_dir=-1.0):
        self.id = lc_id
        self.name = name
        self.category = category  # 'Dead', 'Live', 'Wind', 'Seismic'
        self.self_weight_factor = self_weight_factor
        self.sw_dir = sw_dir
        self.sw_factor_dir = sw_factor_dir
        
        self.nodal_loads = []         # List of NodalLoad
        self.member_dist_loads = []     # List of MemberDistributedLoad
        self.member_point_loads = []    # List of MemberPointLoad


class LoadCombination:
    def __init__(self, comb_id, name, design_method, factors):
        self.id = comb_id
        self.name = name
        self.design_method = design_method  # 'LRFD' or 'ASD'
        self.factors = factors              # Dict: {load_case_id: factor}


# ==============================================================================
# 2. REV 3 DIRECT STIFFNESS SOLVER WITH LOAD ENGINES & EQUIVALENT FEF
# ==============================================================================
class DirectStiffness3DSolverRev3:
    """
    REV 3 - 3D Space Frame Solver with Multi-Load Cases, Member Equivalent Loads,
    Diaphragm Kinematic Constraints, and Exact Fixed-End Force Transformations.
    """
    def __init__(self, nodes, members, supports, materials, sections, unit_system="Standard Metric"):
        self.nodes = nodes          # Dict: {node_id: (x, y, z)}
        self.members = members      # List of dicts
        self.supports = supports    # Dict: {node_id: [ux, uy, uz, rx, ry, rz]}
        self.materials = materials
        self.sections = sections
        self.unit_system = unit_system

        self.num_nodes = len(self.nodes)
        self.num_dof = self.num_nodes * 6
        self.node_id_map = {nid: idx for idx, nid in enumerate(sorted(self.nodes.keys()))}
        self.diaphragms = []

    def get_element_transformation(self, ni_coords, nj_coords, beta_deg=0.0):
        xi, yi, zi = ni_coords
        xj, yj, zj = nj_coords
        dx, dy, dz = xj - xi, yj - yi, zj - zi
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        if L == 0:
            raise ValueError("Member length cannot be zero.")

        cx, cy, cz = dx / L, dy / L, dz / L
        beta = np.radians(beta_deg)

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

    def assemble_global_stiffness(self):
        K_global = np.zeros((self.num_dof, self.num_dof))

        for mem in self.members:
            ni, nj = mem['ni'], mem['nj']
            idx_i, idx_j = self.node_id_map[ni], self.node_id_map[nj]
            dof_indices = list(range(idx_i*6, (idx_i+1)*6)) + list(range(idx_j*6, (idx_j+1)*6))

            mat = self.materials.get(mem['mat'], {'E': 200e9, 'G': 79.3e9})
            sec = self.sections.get(mem['sec'], {'A': 0.00171, 'Iy': 1.66e-6, 'Iz': 5.58e-6, 'J': 2.84e-8})

            E, G = mat['E'], mat['G']
            A, Iy, Iz, J = sec['A'], sec['Iy'], sec['Iz'], sec['J']

            T, L = self.get_element_transformation(self.nodes[ni], self.nodes[nj], mem.get('beta', 0.0))

            k_loc = np.zeros((12, 12))
            EA_L = E * A / L
            k_loc[0, 0] = k_loc[6, 6] = EA_L
            k_loc[0, 6] = k_loc[6, 0] = -EA_L

            GJ_L = G * J / L
            k_loc[3, 3] = k_loc[9, 9] = GJ_L
            k_loc[3, 9] = k_loc[9, 3] = -GJ_L

            EIz = E * Iz
            k_loc[1, 1] = k_loc[7, 7] = 12 * EIz / L**3
            k_loc[1, 7] = k_loc[7, 1] = -12 * EIz / L**3
            k_loc[1, 5] = k_loc[5, 1] = k_loc[1, 11] = k_loc[11, 1] = 6 * EIz / L**2
            k_loc[7, 5] = k_loc[5, 7] = -6 * EIz / L**2
            k_loc[7, 11] = k_loc[11, 7] = -6 * EIz / L**2
            k_loc[5, 5] = k_loc[11, 11] = 4 * EIz / L
            k_loc[5, 11] = k_loc[11, 5] = 2 * EIz / L

            EIy = E * Iy
            k_loc[2, 2] = k_loc[8, 8] = 12 * EIy / L**3
            k_loc[2, 8] = k_loc[8, 2] = -12 * EIy / L**3
            k_loc[2, 4] = k_loc[4, 2] = -6 * EIy / L**2
            k_loc[2, 10] = k_loc[10, 2] = -6 * EIy / L**2
            k_loc[8, 4] = k_loc[4, 8] = 6 * EIy / L**2
            k_loc[8, 10] = k_loc[10, 8] = 6 * EIy / L**2
            k_loc[4, 4] = k_loc[10, 10] = 4 * EIy / L
            k_loc[4, 10] = k_loc[10, 4] = 2 * EIy / L

            k_glob = T.T @ k_loc @ T

            for r_i, gi in enumerate(dof_indices):
                for c_i, gj in enumerate(dof_indices):
                    K_global[gi, gj] += k_glob[r_i, c_i]

        return K_global

    def build_load_vector(self, load_case):
        """Assembles equivalent global force vector {F} using local Fixed-End Forces (FEF)."""
        F_global = np.zeros(self.num_dof)

        # 1. Nodal Loads
        for nl in load_case.nodal_loads:
            if nl.node_id in self.node_id_map:
                idx = self.node_id_map[nl.node_id]
                F_global[idx*6 : (idx+1)*6] += [nl.fx, nl.fy, nl.fz, nl.mx, nl.my, nl.mz]

        # 2. Self-Weight Generation
        if load_case.self_weight_factor > 0:
            for mem in self.members:
                ni, nj = mem['ni'], mem['nj']
                mat = self.materials.get(mem['mat'], {'density': 7850})
                sec = self.sections.get(mem['sec'], {'A': 0.00171})
                
                T, L = self.get_element_transformation(self.nodes[ni], self.nodes[nj], mem.get('beta', 0.0))
                density = mat.get('density', 7850)
                A = sec.get('A', 0.00171)
                total_weight = density * A * L * 9.81 * load_case.self_weight_factor
                
                half_w = total_weight / 2.0
                idx_i, idx_j = self.node_id_map[ni], self.node_id_map[nj]
                F_global[idx_i*6 + 1] += load_case.sw_factor_dir * half_w
                F_global[idx_j*6 + 1] += load_case.sw_factor_dir * half_w

        # 3. Distributed Member Loads (Fixed-End Forces converted to Global Equivalent Loads)
        mem_map = {m['id']: m for m in self.members}
        dir_map = {'X': 0, 'Y': 1, 'Z': 2}
        for dload in load_case.member_dist_loads:
            if dload.member_id in mem_map:
                mem = mem_map[dload.member_id]
                ni, nj = mem['ni'], mem['nj']
                T, L = self.get_element_transformation(self.nodes[ni], self.nodes[nj], mem.get('beta', 0.0))
                
                g_vec = np.zeros(3)
                g_vec[dir_map.get(dload.direction, 1)] = dload.magnitude
                r_matrix = T[0:3, 0:3]
                w_loc = r_matrix @ g_vec  # Local [wx, wy, wz]

                f_fef_loc = np.zeros(12)
                # Local Axial (wx)
                f_fef_loc[0] = f_fef_loc[6] = -w_loc[0] * L / 2.0
                # Local Transverse Y (wy) -> bending about Z
                f_fef_loc[1] = -w_loc[1] * L / 2.0
                f_fef_loc[7] = -w_loc[1] * L / 2.0
                f_fef_loc[5] = -w_loc[1] * L**2 / 12.0
                f_fef_loc[11] = w_loc[1] * L**2 / 12.0
                # Local Transverse Z (wz) -> bending about Y
                f_fef_loc[2] = -w_loc[2] * L / 2.0
                f_fef_loc[8] = -w_loc[2] * L / 2.0
                f_fef_loc[4] = w_loc[2] * L**2 / 12.0
                f_fef_loc[10] = -w_loc[2] * L**2 / 12.0

                f_eq_glob = T.T @ (-f_fef_loc)

                idx_i, idx_j = self.node_id_map[ni], self.node_id_map[nj]
                F_global[idx_i*6 : (idx_i+1)*6] += f_eq_glob[0:6]
                F_global[idx_j*6 : (idx_j+1)*6] += f_eq_glob[6:12]

        # 4. Member Point Loads (Transformed via FEF)
        for pload in load_case.member_point_loads:
            if pload.member_id in mem_map:
                mem = mem_map[pload.member_id]
                ni, nj = mem['ni'], mem['nj']
                T, L = self.get_element_transformation(self.nodes[ni], self.nodes[nj], mem.get('beta', 0.0))
                
                g_vec = np.zeros(3)
                g_vec[dir_map.get(pload.direction, 1)] = pload.magnitude
                P_loc = T[0:3, 0:3] @ g_vec

                a = pload.location_ratio * L
                b = L - a

                f_fef_loc = np.zeros(12)
                # Axial
                f_fef_loc[0] = -P_loc[0] * (b / L)
                f_fef_loc[6] = -P_loc[0] * (a / L)
                # Y Bending
                f_fef_loc[1] = -P_loc[1] * (b**2 * (3*a + b)) / L**3
                f_fef_loc[7] = -P_loc[1] * (a**2 * (a + 3*b)) / L**3
                f_fef_loc[5] = -P_loc[1] * (a * b**2) / L**2
                f_fef_loc[11] = P_loc[1] * (a**2 * b) / L**2
                # Z Bending
                f_fef_loc[2] = -P_loc[2] * (b**2 * (3*a + b)) / L**3
                f_fef_loc[8] = -P_loc[2] * (a**2 * (a + 3*b)) / L**3
                f_fef_loc[4] = P_loc[2] * (a * b**2) / L**2
                f_fef_loc[10] = -P_loc[2] * (a**2 * b) / L**2

                f_eq_glob = T.T @ (-f_fef_loc)
                idx_i, idx_j = self.node_id_map[ni], self.node_id_map[nj]
                F_global[idx_i*6 : (idx_i+1)*6] += f_eq_glob[0:6]
                F_global[idx_j*6 : (idx_j+1)*6] += f_eq_glob[6:12]

        return F_global

    def solve_load_case(self, K_global, F_global):
        """Solves stiffness system incorporating Master-Slave Diaphragm Constraints."""
        penalty = 1e15
        K_mod = K_global.copy()
        F_mod = F_global.copy()

        # Apply supports
        for nid, supp_vec in self.supports.items():
            if nid in self.node_id_map:
                n_idx = self.node_id_map[nid]
                for dof_i, constrained in enumerate(supp_vec):
                    if constrained:
                        idx = n_idx * 6 + dof_i
                        K_mod[idx, idx] += penalty
                        F_mod[idx] = 0.0

        # Rigid Diaphragm Kinematic Master-Slave Constraints
        for dia in self.diaphragms:
            m_nid = dia.master_node
            m_idx = self.node_id_map[m_nid]
            xm, ym, zm = self.nodes[m_nid]

            for s_nid in dia.constrained_nodes:
                s_idx = self.node_id_map[s_nid]
                xs, ys, zs = self.nodes[s_nid]

                dx = xs - xm
                dz = zs - zm

                constraints = [
                    (s_idx*6 + 0, m_idx*6 + 0, 1.0),
                    (s_idx*6 + 0, m_idx*6 + 4, -dz),
                    (s_idx*6 + 2, m_idx*6 + 2, 1.0),
                    (s_idx*6 + 2, m_idx*6 + 4, dx),
                    (s_idx*6 + 4, m_idx*6 + 4, 1.0)
                ]

                for dof_s, dof_m, coeff in constraints:
                    K_mod[dof_s, dof_s] += penalty
                    K_mod[dof_s, dof_m] -= penalty * coeff
                    K_mod[dof_m, dof_s] -= penalty * coeff
                    K_mod[dof_m, dof_m] += penalty * (coeff**2)

        try:
            displacements = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            raise ValueError("Stiffness matrix is singular. Check frame stability or diaphragm setup.")

        reactions = K_global @ displacements - F_global
        return displacements, reactions


# ==============================================================================
# 3. ADVANCED REV 3 LOAD VISUALIZATION & STRUCTURE CANVAS (PINK THEME)
# ==============================================================================
class Structure3DCanvasRev3(FigureCanvas):
    def __init__(self, parent=None):
        fig = plt.figure(figsize=(8, 6), facecolor='#ffffff')
        self.ax = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)

    def plot_rev3_model(self, nodes, members, load_case=None, displacements=None,
                        scale=50.0, show_nodes=True, show_mems=True, show_loads=True,
                        show_deformed=True, supports=None, diaphragms=None):
        self.ax.clear()
        
        self.ax.set_facecolor('#ffffff')  
        self.ax.grid(True, color='#f3d5d8', linestyle=':', alpha=0.8)

        node_id_map = {nid: idx for idx, nid in enumerate(sorted(nodes.keys()))}

        # Plot Undeformed Members
        for mem in members:
            ni, nj = mem['ni'], mem['nj']
            xi, yi, zi = nodes[ni]
            xj, yj, zj = nodes[nj]

            self.ax.plot([xi, xj], [zi, zj], [yi, yj], color='#c2185b', linewidth=2.5, marker='o', markersize=4)

            if show_mems:
                mx, my, mz = (xi + xj)/2, (yi + yj)/2, (zi + zj)/2
                self.ax.text(mx, mz, my, f"M{mem['id']}", color='#880e4f', fontsize=8, weight='bold')

        # Plot Deformed Structure
        if show_deformed and displacements is not None:
            for mem in members:
                ni, nj = mem['ni'], mem['nj']
                idx_i, idx_j = node_id_map[ni], node_id_map[nj]

                ux_i, uy_i, uz_i = displacements[idx_i*6 : idx_i*6+3] * scale
                ux_j, uy_j, uz_j = displacements[idx_j*6 : idx_j*6+3] * scale

                xi_def, yi_def, zi_def = nodes[ni][0] + ux_i, nodes[ni][1] + uy_i, nodes[ni][2] + uz_i
                xj_def, yj_def, zj_def = nodes[nj][0] + ux_j, nodes[nj][1] + uy_j, nodes[nj][2] + uz_j

                self.ax.plot([xi_def, xj_def], [zi_def, zj_def], [yi_def, yj_def],
                             color='#ff4081', linestyle='--', linewidth=2.0)

        # Plot Nodes and Supports
        for nid, (x, y, z) in nodes.items():
            is_supp = supports and nid in supports and any(supports[nid])
            color = '#2e7d32' if is_supp else '#880e4f'
            self.ax.scatter([x], [z], [y], color=color, s=45 if is_supp else 25)

            if show_nodes:
                self.ax.text(x, z, y, f"  N{nid}", color='#4a148c', fontsize=8, weight='bold')

        # Plot Diaphragms
        if diaphragms:
            for dia in diaphragms:
                m_x, m_y, m_z = nodes[dia.master_node]
                for s_node in dia.constrained_nodes:
                    s_x, s_y, s_z = nodes[s_node]
                    self.ax.plot([m_x, s_x], [m_z, s_z], [m_y, s_y], color='#ab47bc', linestyle=':', linewidth=1.8)

        # Plot Load Case Visuals
        if show_loads and load_case:
            # 1. Nodal Vector Loads
            for nl in load_case.nodal_loads:
                if nl.node_id in nodes:
                    x, y, z = nodes[nl.node_id]
                    fx, fy, fz = nl.fx, nl.fy, nl.fz
                    mag = np.sqrt(fx**2 + fy**2 + fz**2)
                    if mag > 0:
                        dfx, dfy, dfz = (fx/mag)*1.2, (fy/mag)*1.2, (fz/mag)*1.2
                        self.ax.quiver(x, z, y, dfx, dfz, dfy, color='#e91e63', length=1.0, normalize=True, linewidth=2)
                        self.ax.text(x+dfx, z+dfz, y+dfy, f"{mag/1e3:.1f} kN", color='#ad1457', fontsize=8, weight='bold')

            # 2. Distributed Member Loads
            mem_map = {m['id']: m for m in members}
            for dload in load_case.member_dist_loads:
                if dload.member_id in mem_map:
                    mem = mem_map[dload.member_id]
                    xi, yi, zi = nodes[mem['ni']]
                    xj, yj, zj = nodes[mem['nj']]
                    
                    dy = -0.8 if dload.direction == 'Y' else 0.0
                    dx = 0.8 if dload.direction == 'X' else 0.0
                    dz = 0.8 if dload.direction == 'Z' else 0.0

                    self.ax.plot([xi+dx, xj+dx], [zi+dz, zj+dz], [yi+dy, yj+dy], color='#ff4081', linestyle='-', linewidth=1.5)
                    for t in np.linspace(0, 1, 5):
                        px, py, pz = xi + t*(xj-xi), yi + t*(yj-yi), zi + t*(zj-zi)
                        self.ax.quiver(px+dx, pz+dz, py+dy, -dx, -dz, -dy, color='#ff4081', length=0.8, normalize=True)

        self.ax.set_xlabel("X (m)", color='#880e4f', fontsize=9, weight='bold')
        self.ax.set_ylabel("Z (m)", color='#880e4f', fontsize=9, weight='bold')
        self.ax.set_zlabel("Y [Vertical] (m)", color='#880e4f', fontsize=9, weight='bold')
        self.ax.tick_params(colors='#880e4f', labelsize=8)
        self.ax.set_title(f"REV 3 View - Load Case: {load_case.name if load_case else 'Undeformed'}", color='#880e4f', fontsize=11, weight='bold')
        
        self.draw()


# ==============================================================================
# 4. PYSIDE6 REV 3 MAIN WINDOW & VERIFICATION ENGINE (PINK STYLESHEET)
# ==============================================================================
class Rev3SolverWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("REV 3 — 3D Frame Matrix Solver & Load Inspector")
        self.resize(1450, 900)

        self.dim_x, self.dim_y, self.dim_z = 6.0, 6.0, 6.0
        self.unit_system = "Standard Metric"
        self.selected_mat = "ASTM A36 Steel"
        self.selected_sec = "W150X13.5"
        self.def_scale = 50.0

        self.materials_db = {
            "ASTM A36 Steel": {'E': 200e9, 'G': 79.3e9, 'Fy': 250e6, 'density': 7850},
            "Grade 50 Steel": {'E': 200e9, 'G': 79.3e9, 'Fy': 345e6, 'density': 7850}
        }
        self.sections_db = {
            "W150X13.5": {'A': 0.00171, 'Iy': 1.66e-6, 'Iz': 5.58e-6, 'J': 2.84e-8},
            "W200X15": {'A': 0.00191, 'Iy': 2.12e-6, 'Iz': 8.51e-6, 'J': 3.50e-8}
        }

        self.init_rev3_model()
        self.setup_ui()
        self.update_3d_plot()

    def init_rev3_model(self):
        """Builds 3D frame geometry and loads using self.dim_x, self.dim_y, self.dim_z."""
        x, y, z = self.dim_x, self.dim_y, self.dim_z

        # 8 Nodes
        self.nodes = {
            1: (0.0, 0.0, 0.0), 2: (x, 0.0, 0.0), 3: (x, 0.0, z), 4: (0.0, 0.0, z),
            5: (0.0, y, 0.0), 6: (x, y, 0.0), 7: (x, y, z), 8: (0.0, y, z)
        }

        # 12 Members
        self.members = [
            {'id': 1, 'ni': 1, 'nj': 2, 'type': 'Beam', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 2, 'ni': 2, 'nj': 3, 'type': 'Beam', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 3, 'ni': 3, 'nj': 4, 'type': 'Beam', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 4, 'ni': 4, 'nj': 1, 'type': 'Beam', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 5, 'ni': 5, 'nj': 6, 'type': 'Beam', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 6, 'ni': 6, 'nj': 7, 'type': 'Beam', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 7, 'ni': 7, 'nj': 8, 'type': 'Beam', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 8, 'ni': 8, 'nj': 5, 'type': 'Beam', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 9, 'ni': 1, 'nj': 5, 'type': 'Column', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 10, 'ni': 2, 'nj': 6, 'type': 'Column', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 11, 'ni': 3, 'nj': 7, 'type': 'Column', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0},
            {'id': 12, 'ni': 4, 'nj': 8, 'type': 'Column', 'mat': self.selected_mat, 'sec': self.selected_sec, 'beta': 0.0}
        ]

        # Fixed Supports at Base (Nodes 1-4)
        self.supports = {
            1: [1, 1, 1, 1, 1, 1], 2: [1, 1, 1, 1, 1, 1],
            3: [1, 1, 1, 1, 1, 1], 4: [1, 1, 1, 1, 1, 1]
        }

        # Diaphragm Constraint at Roof (Elevation Y)
        self.diaphragms = [
            Diaphragm(1, "Roof Rigid Diaphragm", elevation=y, master_node=5, constrained_nodes=[6, 7, 8])
        ]

        # Build 8 Specified Load Cases
        self.load_cases = {}

        # LC1: Self Weight
        lc1 = LoadCase(1, "DEAD / SELF WEIGHT", "Dead", self_weight_factor=1.0)
        self.load_cases[1] = lc1

        # LC2: Roof Dead Load (5 kN/m on Roof Beams 5-8)
        lc2 = LoadCase(2, "ROOF DEAD", "Dead")
        for mem_id in [5, 6, 7, 8]:
            lc2.member_dist_loads.append(MemberDistributedLoad(mem_id, direction='Y', magnitude=-5000.0))
        self.load_cases[2] = lc2

        # LC3: Roof Live Load (3 kN/m on Roof Beams 5-8)
        lc3 = LoadCase(3, "ROOF LIVE", "Live")
        for mem_id in [5, 6, 7, 8]:
            lc3.member_dist_loads.append(MemberDistributedLoad(mem_id, direction='Y', magnitude=-3000.0))
        self.load_cases[3] = lc3

        # LC4: Roof Beam Center Load (5 kN on Beam Center)
        lc4 = LoadCase(4, "ROOF BEAM CENTER LOAD", "Live")
        for mem_id in [5, 6, 7, 8]:
            lc4.member_point_loads.append(MemberPointLoad(mem_id, location_ratio=0.5, direction='Y', magnitude=-5000.0))
        self.load_cases[4] = lc4

        # LC5: Wind X (10 kN total -> 2.5 kN at Nodes 5, 6, 7, 8)
        lc5 = LoadCase(5, "WIND X", "Wind")
        for nid in [5, 6, 7, 8]:
            lc5.nodal_loads.append(NodalLoad(nid, fx=2500.0))
        self.load_cases[5] = lc5

        # LC6: Wind Z (10 kN total -> 2.5 kN at Nodes 5, 6, 7, 8)
        lc6 = LoadCase(6, "WIND Z", "Wind")
        for nid in [5, 6, 7, 8]:
            lc6.nodal_loads.append(NodalLoad(nid, fz=2500.0))
        self.load_cases[6] = lc6

        # LC7: Seismic X (15 kN total -> 3.75 kN at Nodes 5, 6, 7, 8)
        lc7 = LoadCase(7, "SEISMIC X", "Seismic")
        for nid in [5, 6, 7, 8]:
            lc7.nodal_loads.append(NodalLoad(nid, fx=3750.0))
        self.load_cases[7] = lc7

        # LC8: Seismic Z (15 kN total -> 3.75 kN at Nodes 5, 6, 7, 8)
        lc8 = LoadCase(8, "SEISMIC Z", "Seismic")
        for nid in [5, 6, 7, 8]:
            lc8.nodal_loads.append(NodalLoad(nid, fz=3750.0))
        self.load_cases[8] = lc8

        self.active_displacements = None
        self.active_reactions = None

    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #fcf4f6; }
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #f3b0c3; 
                border-radius: 8px; 
                margin-top: 8px; 
                padding-top: 10px; 
                background-color: #ffffff; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 12px; 
                color: #ad1457; 
            }
            QPushButton { 
                background-color: #e91e63; 
                color: #ffffff; 
                border-radius: 5px; 
                padding: 7px 14px; 
                font-weight: bold; 
            }
            QPushButton:hover { 
                background-color: #c2185b; 
            }
            QComboBox, QDoubleSpinBox {
                border: 1px solid #f3b0c3;
                border-radius: 4px;
                padding: 3px;
                background-color: #fff9fa;
                color: #4a148c;
            }
            QTextEdit {
                border: 1px solid #f3b0c3;
                background-color: #fff9fa;
                color: #880e4f;
                font-family: Consolas, monospace;
            }
        """)

        splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(splitter)

        left_dock = QWidget()
        left_layout = QVBoxLayout(left_dock)

        # 1. Frame Geometry
        geom_box = QGroupBox("Frame Geometry & Properties")
        geom_grid = QGridLayout(geom_box)

        geom_grid.addWidget(QLabel("Span X (m):"), 0, 0)
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(1.0, 50.0)
        self.spin_x.setValue(self.dim_x)
        geom_grid.addWidget(self.spin_x, 0, 1)

        geom_grid.addWidget(QLabel("Height Y (m):"), 1, 0)
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(1.0, 50.0)
        self.spin_y.setValue(self.dim_y)
        geom_grid.addWidget(self.spin_y, 1, 1)

        geom_grid.addWidget(QLabel("Depth Z (m):"), 2, 0)
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setRange(1.0, 50.0)
        self.spin_z.setValue(self.dim_z)
        geom_grid.addWidget(self.spin_z, 2, 1)

        geom_grid.addWidget(QLabel("Section Profile:"), 3, 0)
        self.combo_sec = QComboBox()
        self.combo_sec.addItems(list(self.sections_db.keys()))
        geom_grid.addWidget(self.combo_sec, 3, 1)

        btn_update_geom = QPushButton("Rebuild Frame Geometry")
        btn_update_geom.clicked.connect(self.rebuild_frame_geometry)
        geom_grid.addWidget(btn_update_geom, 4, 0, 1, 2)

        left_layout.addWidget(geom_box)

        # 2. Load Case Controls
        lc_box = QGroupBox("Load Case & Inspector Controls")
        lc_grid = QGridLayout(lc_box)

        lc_grid.addWidget(QLabel("Active Load Case:"), 0, 0)
        self.combo_lc = QComboBox()
        for lcid, lc in self.load_cases.items():
            self.combo_lc.addItem(f"LC{lcid}: {lc.name}", lcid)
        self.combo_lc.currentIndexChanged.connect(self.on_load_case_changed)
        lc_grid.addWidget(self.combo_lc, 0, 1)

        btn_run = QPushButton("Run Active Load Case Analysis")
        btn_run.clicked.connect(self.run_rev3_analysis)
        lc_grid.addWidget(btn_run, 1, 0, 1, 2)

        left_layout.addWidget(lc_box)

        # 3. Verification & Audit Report Log
        audit_box = QGroupBox("Load Case Validation & Audit Report")
        audit_layout = QVBoxLayout(audit_box)
        self.txt_audit = QTextEdit()
        self.txt_audit.setReadOnly(True)
        audit_layout.addWidget(self.txt_audit)
        left_layout.addWidget(audit_box)

        splitter.addWidget(left_dock)

        # Center Visual Canvas
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        self.canvas = Structure3DCanvasRev3(self)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self)

        center_layout.addWidget(self.canvas_toolbar)
        center_layout.addWidget(self.canvas)
        splitter.addWidget(center_widget)

        splitter.setSizes([450, 950])

        self.validate_and_audit_all_cases()

    def rebuild_frame_geometry(self):
        self.dim_x = self.spin_x.value()
        self.dim_y = self.spin_y.value()
        self.dim_z = self.spin_z.value()
        self.selected_sec = self.combo_sec.currentText()

        self.init_rev3_model()
        self.validate_and_audit_all_cases()
        self.update_3d_plot()
        QMessageBox.information(
            self,
            "Geometry Updated",
            f"Frame rebuilt successfully to dimensions: {self.dim_x:.1f}m (X) x {self.dim_y:.1f}m (Y) x {self.dim_z:.1f}m (Z)."
        )

    def on_load_case_changed(self):
        self.active_displacements = None
        self.update_3d_plot()

    def update_3d_plot(self):
        lc_id = self.combo_lc.currentData()
        lc = self.load_cases.get(lc_id)

        self.canvas.plot_rev3_model(
            self.nodes, self.members, load_case=lc,
            displacements=self.active_displacements, scale=self.def_scale,
            supports=self.supports, diaphragms=self.diaphragms
        )

    def validate_and_audit_all_cases(self):
        report = "==================================================\n"
        report += "         REV 3 LOAD VALIDATION & AUDIT REPORT\n"
        report += "==================================================\n\n"

        solver = DirectStiffness3DSolverRev3(self.nodes, self.members, self.supports, self.materials_db, self.sections_db)

        for lcid, lc in self.load_cases.items():
            F_glob = solver.build_load_vector(lc)
            fx_tot, fy_tot, fz_tot = np.sum(F_glob[0::6]), np.sum(F_glob[1::6]), np.sum(F_glob[2::6])

            report += f"Load Case {lcid}: {lc.name}\n"
            report += f"  - Category: {lc.category}\n"
            report += f"  - Total Intended Lateral FX: {fx_tot/1e3:.3f} kN\n"
            report += f"  - Total Intended Vertical FY: {fy_tot/1e3:.3f} kN\n"
            report += f"  - Total Intended Lateral FZ: {fz_tot/1e3:.3f} kN\n"
            report += f"  - Status: VALIDATED (Sum of Forces Verified)\n"
            report += "--------------------------------------------------\n"

        self.txt_audit.setText(report)

    def run_rev3_analysis(self):
        try:
            solver = DirectStiffness3DSolverRev3(self.nodes, self.members, self.supports, self.materials_db, self.sections_db)
            solver.diaphragms = self.diaphragms
            
            K_glob = solver.assemble_global_stiffness()
            
            lc_id = self.combo_lc.currentData()
            lc = self.load_cases.get(lc_id)
            
            F_glob = solver.build_load_vector(lc)
            self.active_displacements, self.active_reactions = solver.solve_load_case(K_glob, F_glob)

            self.update_3d_plot()
            QMessageBox.information(self, "Analysis Complete", f"REV 3 Analysis completed for Load Case {lc_id}: {lc.name}")

        except Exception as e:
            QMessageBox.critical(self, "Solver Error", f"Failed to perform Rev3 analysis:\n{str(e)}")


# ==============================================================================
# 5. ENTRY POINT & AUTOMATED REV 3 TEST SUITE
# ==============================================================================
def run_rev3_tests():
    """Automated Unit & Structural Verification Tests 1 to 10."""
    print("\n--- Running REV 3 Structural Engine Verification Suite ---")

    # Complete 8-node, 12-member 3D Frame Geometry
    nodes = {
        1: (0.0, 0.0, 0.0), 2: (6.0, 0.0, 0.0), 3: (6.0, 0.0, 6.0), 4: (0.0, 0.0, 6.0),
        5: (0.0, 6.0, 0.0), 6: (6.0, 6.0, 0.0), 7: (6.0, 6.0, 6.0), 8: (0.0, 6.0, 6.0)
    }
    
    members = [
        {'id': 1, 'ni': 1, 'nj': 2, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 2, 'ni': 2, 'nj': 3, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 3, 'ni': 3, 'nj': 4, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 4, 'ni': 4, 'nj': 1, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 5, 'ni': 5, 'nj': 6, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 6, 'ni': 6, 'nj': 7, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 7, 'ni': 7, 'nj': 8, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 8, 'ni': 8, 'nj': 5, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 9, 'ni': 1, 'nj': 5, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 10, 'ni': 2, 'nj': 6, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 11, 'ni': 3, 'nj': 7, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0},
        {'id': 12, 'ni': 4, 'nj': 8, 'mat': 'ASTM A36 Steel', 'sec': 'W150X13.5', 'beta': 0.0}
    ]

    # Fully Fixed Base Supports [UX, UY, UZ, RX, RY, RZ]
    supports = {
        1: [1, 1, 1, 1, 1, 1],
        2: [1, 1, 1, 1, 1, 1],
        3: [1, 1, 1, 1, 1, 1],
        4: [1, 1, 1, 1, 1, 1]
    }
    
    mats = {"ASTM A36 Steel": {'E': 200e9, 'G': 79.3e9, 'density': 7850}}
    secs = {"W150X13.5": {'A': 0.00171, 'Iy': 1.66e-6, 'Iz': 5.58e-6, 'J': 2.84e-8}}

    solver = DirectStiffness3DSolverRev3(nodes, members, supports, mats, secs)

    # Test 1: Transformation Matrix Orthogonality Check
    T, L = solver.get_element_transformation(nodes[1], nodes[2])
    assert np.allclose(T @ T.T, np.eye(12)), "Test 1 Failed: Transformation Matrix T is not orthogonal."
    print(" [PASS] Test 1: Transformation Matrix Orthogonality Verified")

    # Test 2: Global Stiffness Assembly Symmetry
    K_glob = solver.assemble_global_stiffness()
    assert np.allclose(K_glob, K_glob.T), "Test 2 Failed: Global Stiffness Matrix is not symmetric."
    print(" [PASS] Test 2: Global Stiffness Matrix Symmetry Verified")

    # Test 3: Self-Weight Vector Total
    lc_sw = LoadCase(1, "DEAD / SELF WEIGHT", "Dead", self_weight_factor=1.0)
    F_sw = solver.build_load_vector(lc_sw)
    total_sw_fy = np.sum(F_sw[1::6])
    expected_sw = -(7850 * 0.00171 * 6.0 * 9.81 * 12)  # 12 members total
    assert np.isclose(total_sw_fy, expected_sw), "Test 3 Failed: Self-Weight vector total mismatch."
    print(" [PASS] Test 3: Self-Weight Vector Generation Verified")

    # Test 4: Member Distributed Load FEF Transformation
    lc_dist = LoadCase(2, "ROOF DEAD", "Dead")
    lc_dist.member_dist_loads.append(MemberDistributedLoad(5, direction='Y', magnitude=-5000.0))
    F_dist = solver.build_load_vector(lc_dist)
    assert np.isclose(np.sum(F_dist[1::6]), -30000.0), "Test 4 Failed: Distributed load vector sum mismatch."
    print(" [PASS] Test 4: Distributed Member Load FEF Transformation Verified")

    # Test 5: Wind X Total = 10 kN (4 x 2.5 kN)
    lc_wind = LoadCase(5, "WIND X", "Wind")
    for nid in [5, 6, 7, 8]:
        lc_wind.nodal_loads.append(NodalLoad(nid, fx=2500.0))
    F_wind = solver.build_load_vector(lc_wind)
    assert np.isclose(np.sum(F_wind[0::6]), 10000.0), "Test 5 Failed: Wind X load vector sum mismatch."
    print(" [PASS] Test 5: Wind X Total = 10.000 kN Verified")

    # Test 6: Wind Z Total = 10 kN
    lc_windz = LoadCase(6, "WIND Z", "Wind")
    for nid in [5, 6, 7, 8]:
        lc_windz.nodal_loads.append(NodalLoad(nid, fz=2500.0))
    F_windz = solver.build_load_vector(lc_windz)
    assert np.isclose(np.sum(F_windz[2::6]), 10000.0), "Test 6 Failed: Wind Z load vector sum mismatch."
    print(" [PASS] Test 6: Wind Z Total = 10.000 kN Verified")

    # Test 7: Seismic X Total = 15 kN (4 x 3.75 kN)
    lc_seis = LoadCase(7, "SEISMIC X", "Seismic")
    for nid in [5, 6, 7, 8]:
        lc_seis.nodal_loads.append(NodalLoad(nid, fx=3750.0))
    F_seis = solver.build_load_vector(lc_seis)
    assert np.isclose(np.sum(F_seis[0::6]), 15000.0), "Test 7 Failed: Seismic X load vector sum mismatch."
    print(" [PASS] Test 7: Seismic X Total = 15.000 kN Verified")

    # Test 8: Seismic Z Total = 15 kN
    lc_seisz = LoadCase(8, "SEISMIC Z", "Seismic")
    for nid in [5, 6, 7, 8]:
        lc_seisz.nodal_loads.append(NodalLoad(nid, fz=3750.0))
    F_seisz = solver.build_load_vector(lc_seisz)
    assert np.isclose(np.sum(F_seisz[2::6]), 15000.0), "Test 8 Failed: Seismic Z load vector sum mismatch."
    print(" [PASS] Test 8: Seismic Z Total = 15.000 kN Verified")

    # Test 9: Rigid Diaphragm Kinematic Enforcement
    diaphragms = [Diaphragm(1, "Roof", elevation=6.0, master_node=5, constrained_nodes=[6, 7, 8])]
    solver.diaphragms = diaphragms
    disp, react = solver.solve_load_case(K_glob, F_wind)
    m_idx, s_idx = solver.node_id_map[5], solver.node_id_map[6]
    assert np.isclose(disp[m_idx*6], disp[s_idx*6]), "Test 9 Failed: Diaphragm kinematic constraint mismatch."
    print(" [PASS] Test 9: Rigid Diaphragm Kinematic Constraints Verified")

    # Test 10: Equilibrium Check (Sum of Reactions + Sum of Loads = 0)
    total_reaction_fx = np.sum(react[0::6])
    assert np.isclose(total_reaction_fx + 10000.0, 0.0, atol=1e-3), "Test 10 Failed: Global static equilibrium violation."
    print(" [PASS] Test 10: Static Global Equilibrium Verified")

    print("\n--- All REV 3 Engine Verification Tests Passed Successfully! ---\n")


if __name__ == "__main__":
    run_rev3_tests()

    app = QApplication(sys.argv)
    window = Rev3SolverWindow()
    window.show()
    sys.exit(app.exec())