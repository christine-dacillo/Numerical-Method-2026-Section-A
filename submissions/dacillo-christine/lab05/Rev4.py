import sys
import os
import numpy as np
import pandas as pd

# PyQt6 Qt GUI Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QDockWidget, QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton,
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox, QToolBar,
    QFileDialog, QMessageBox, QHeaderView, QSplitter, QFrame, QTextEdit, QScrollArea,
    QStatusBar, QMenuBar, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QFont, QAction, QColor

# Matplotlib Qt Integration
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


# ==============================================================================
# 1. LOAD & DIAPHRAGM DATA STRUCTURES
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
    def __init__(self, member_id, direction='Y', magnitude=0.0):
        self.member_id = member_id
        self.direction = direction.upper()  # 'X', 'Y', 'Z'
        self.magnitude = magnitude          # N/m


class MemberPointLoad:
    def __init__(self, member_id, location_ratio=0.5, direction='Y', magnitude=0.0):
        self.member_id = member_id
        self.location_ratio = location_ratio
        self.direction = direction.upper()
        self.magnitude = magnitude            # N


class Diaphragm:
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
        self.category = category
        self.self_weight_factor = self_weight_factor
        self.sw_dir = sw_dir
        self.sw_factor_dir = sw_factor_dir

        self.nodal_loads = []
        self.member_dist_loads = []
        self.member_point_loads = []


# ==============================================================================
# 2. STRUCTURAL ANALYSIS ENGINE (DIRECT STIFFNESS METHOD)
# ==============================================================================
class DirectStiffness3DSolver:
    def __init__(self, nodes, members, supports, loads, materials, sections,
                 unit_system="Standard Metric", diaphragms=None):
        self.nodes = nodes          # Dict: {node_id: (x, y, z)}
        self.members = members      # List of dicts
        self.supports = supports    # Dict: {node_id: [ux, uy, uz, rx, ry, rz]}
        self.loads = loads or {}
        self.materials = materials
        self.sections = sections
        self.unit_system = unit_system
        self.diaphragms = diaphragms if diaphragms is not None else []

        self.num_nodes = len(self.nodes)
        self.num_dof = self.num_nodes * 6
        self.node_id_map = {nid: idx for idx, nid in enumerate(sorted(self.nodes.keys()))}
        self.K_global = None
        self.F_global = None
        self.displacements = None
        self.reactions = None

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

            # Axial
            EA_L = E * A / L
            k_loc[0, 0] = k_loc[6, 6] = EA_L
            k_loc[0, 6] = k_loc[6, 0] = -EA_L

            # Torsion
            GJ_L = G * J / L
            k_loc[3, 3] = k_loc[9, 9] = GJ_L
            k_loc[3, 9] = k_loc[9, 3] = -GJ_L

            # Bending about local z
            EIz = E * Iz
            k_loc[1, 1] = k_loc[7, 7] = 12 * EIz / L**3
            k_loc[1, 7] = k_loc[7, 1] = -12 * EIz / L**3
            k_loc[1, 5] = k_loc[5, 1] = k_loc[1, 11] = k_loc[11, 1] = 6 * EIz / L**2
            k_loc[7, 5] = k_loc[5, 7] = -6 * EIz / L**2
            k_loc[7, 11] = k_loc[11, 7] = -6 * EIz / L**2
            k_loc[5, 5] = k_loc[11, 11] = 4 * EIz / L
            k_loc[5, 11] = k_loc[11, 5] = 2 * EIz / L

            # Bending about local y
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
        F_global = np.zeros(self.num_dof)

        for nl in load_case.nodal_loads:
            if nl.node_id in self.node_id_map:
                idx = self.node_id_map[nl.node_id]
                F_global[idx*6 : (idx+1)*6] += [nl.fx, nl.fy, nl.fz, nl.mx, nl.my, nl.mz]

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
                w_loc = r_matrix @ g_vec

                f_fef_loc = np.zeros(12)
                f_fef_loc[0] = f_fef_loc[6] = -w_loc[0] * L / 2.0
                f_fef_loc[1] = -w_loc[1] * L / 2.0
                f_fef_loc[7] = -w_loc[1] * L / 2.0
                f_fef_loc[5] = -w_loc[1] * L**2 / 12.0
                f_fef_loc[11] = w_loc[1] * L**2 / 12.0
                f_fef_loc[2] = -w_loc[2] * L / 2.0
                f_fef_loc[8] = -w_loc[2] * L / 2.0
                f_fef_loc[4] = w_loc[2] * L**2 / 12.0
                f_fef_loc[10] = -w_loc[2] * L**2 / 12.0

                f_eq_glob = T.T @ (-f_fef_loc)

                idx_i, idx_j = self.node_id_map[ni], self.node_id_map[nj]
                F_global[idx_i*6 : (idx_i+1)*6] += f_eq_glob[0:6]
                F_global[idx_j*6 : (idx_j+1)*6] += f_eq_glob[6:12]

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
                f_fef_loc[0] = -P_loc[0] * (b / L)
                f_fef_loc[6] = -P_loc[0] * (a / L)
                f_fef_loc[1] = -P_loc[1] * (b**2 * (3*a + b)) / L**3
                f_fef_loc[7] = -P_loc[1] * (a**2 * (a + 3*b)) / L**3
                f_fef_loc[5] = -P_loc[1] * (a * b**2) / L**2
                f_fef_loc[11] = P_loc[1] * (a**2 * b) / L**2
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
        penalty = 1e15
        K_mod = K_global.copy()
        F_mod = F_global.copy()

        for nid, supp_vec in self.supports.items():
            if nid in self.node_id_map:
                n_idx = self.node_id_map[nid]
                for dof_i, constrained in enumerate(supp_vec):
                    if constrained:
                        idx = n_idx * 6 + dof_i
                        K_mod[idx, idx] += penalty
                        F_mod[idx] = 0.0

        for dia in self.diaphragms:
            m_nid = dia.master_node
            m_idx = self.node_id_map[m_nid]
            xm, ym, zm = self.nodes[m_nid]

            for s_nid in dia.constrained_nodes:
                s_idx = self.node_id_map[s_nid]
                xs, ys, zs = self.nodes[s_nid]

                dx = xs - xm
                dz = zs - zm

                constraint_rows = [
                    [(s_idx*6 + 0, 1.0), (m_idx*6 + 0, -1.0), (m_idx*6 + 4, -dz)],
                    [(s_idx*6 + 2, 1.0), (m_idx*6 + 2, -1.0), (m_idx*6 + 4, dx)],
                    [(s_idx*6 + 4, 1.0), (m_idx*6 + 4, -1.0)]
                ]

                for row in constraint_rows:
                    idxs = [item[0] for item in row]
                    g = np.array([item[1] for item in row])
                    K_mod[np.ix_(idxs, idxs)] += penalty * np.outer(g, g)

        try:
            displacements = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            raise ValueError("Stiffness matrix is singular. Ensure structure has proper boundary supports.")

        reactions = K_global @ displacements - F_global
        return displacements, reactions


# ==============================================================================
# 3. EMBEDDED MATPLOTLIB 3D CANVAS
# ==============================================================================
class Structure3DCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = plt.figure(figsize=(8, 6), facecolor='#fff0f5')
        self.ax = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)

    def plot_structure(self, nodes, members, displacements=None, scale=50.0,
                       show_node_labels=True, show_mem_labels=True,
                       show_loads=True, show_deformed=True, show_diaphragms=True,
                       load_case=None, supports=None, diaphragms=None, unit_system="Standard Metric"):
        self.ax.clear()
        self.ax.set_facecolor("#fff0f5")
        self.ax.grid(True, color='#e0b0be', linestyle=':', alpha=0.6)

        node_id_map = {nid: idx for idx, nid in enumerate(sorted(nodes.keys()))}
        mem_map = {m['id']: m for m in members}

        is_imperial = (unit_system == "Imperial")
        len_unit = "ft" if is_imperial else "m"
        force_unit = "kips" if is_imperial else "kN"
        len_mult = 3.28084 if is_imperial else 1.0
        force_mult = 0.000224809 if is_imperial else 0.001

        # Plot Undeformed Structure
        for mem in members:
            ni, nj = mem['ni'], mem['nj']
            xi, yi, zi = [c * len_mult for c in nodes[ni]]
            xj, yj, zj = [c * len_mult for c in nodes[nj]]

            self.ax.plot([xi, xj], [zi, zj], [yi, yj], color='#b83280', linewidth=2.5, marker='o', markersize=4)

            if show_mem_labels:
                mx, my, mz = (xi + xj)/2, (yi + yj)/2, (zi + zj)/2
                self.ax.text(mx, mz, my, f" M{mem['id']}", color='#8c1d58', fontsize=8, weight='bold')

        # Plot Deformed Structure
        if show_deformed and displacements is not None:
            for mem in members:
                ni, nj = mem['ni'], mem['nj']
                idx_i, idx_j = node_id_map[ni], node_id_map[nj]

                ux_i, uy_i, uz_i = displacements[idx_i*6 : idx_i*6+3] * scale * len_mult
                ux_j, uy_j, uz_j = displacements[idx_j*6 : idx_j*6+3] * scale * len_mult

                xi_def = nodes[ni][0] * len_mult + ux_i
                yi_def = nodes[ni][1] * len_mult + uy_i
                zi_def = nodes[ni][2] * len_mult + uz_i

                xj_def = nodes[nj][0] * len_mult + ux_j
                yj_def = nodes[nj][1] * len_mult + uy_j
                zj_def = nodes[nj][2] * len_mult + uz_j

                self.ax.plot([xi_def, xj_def], [zi_def, zj_def], [yi_def, yj_def],
                             color='#e05638', linestyle='--', linewidth=2.2)

        # Plot Nodes and Supports
        for nid, coords in nodes.items():
            x, y, z = [c * len_mult for c in coords]
            is_supported = supports and nid in supports and any(supports[nid])
            marker_color = '#701a45' if is_supported else '#d64d8a'
            self.ax.scatter([x], [z], [y], color=marker_color, s=40 if is_supported else 25)

            if show_node_labels:
                dof_start = (nid - 1) * 6 + 1
                label_txt = f"  N{nid}\n  DOF {dof_start}-{dof_start+5}"
                self.ax.text(x, z, y, label_txt, color='#4a0e2e', fontsize=7, weight='bold')

        # Plot Rigid Diaphragm Links
        if show_diaphragms and diaphragms:
            for dia in diaphragms:
                m_x, m_y, m_z = [c * len_mult for c in nodes[dia.master_node]]
                for s_node in dia.constrained_nodes:
                    s_x, s_y, s_z = [c * len_mult for c in nodes[s_node]]
                    self.ax.plot([m_x, s_x], [m_z, s_z], [m_y, s_y], color='#a855f7', linestyle=':', linewidth=1.8)

        # Plot Active Loads
        if show_loads and load_case:
            load_color = '#c53030'
            axis_vec = {'X': (1.0, 0.0, 0.0), 'Y': (0.0, 1.0, 0.0), 'Z': (0.0, 0.0, 1.0)}

            for nl in load_case.nodal_loads:
                if nl.node_id in nodes:
                    x, y, z = [c * len_mult for c in nodes[nl.node_id]]
                    fx, fy, fz = nl.fx, nl.fy, nl.fz
                    mag = np.sqrt(fx**2 + fy**2 + fz**2)
                    if mag > 0:
                        dfx, dfy, dfz = (fx/mag)*1.2, (fy/mag)*1.2, (fz/mag)*1.2
                        self.ax.quiver(x, z, y, dfx, dfz, dfy, color=load_color, length=1.0, normalize=True, linewidth=2)
                        self.ax.text(x + dfx, z + dfz, y + dfy, f" {mag*force_mult:.1f} {force_unit}", color=load_color, fontsize=8, weight='bold')

            for dload in load_case.member_dist_loads:
                if dload.member_id in mem_map and dload.magnitude != 0:
                    mem = mem_map[dload.member_id]
                    xi, yi, zi = [c * len_mult for c in nodes[mem['ni']]]
                    xj, yj, zj = [c * len_mult for c in nodes[mem['nj']]]

                    sgn = 1.0 if dload.magnitude > 0 else -1.0
                    ux, uy, uz = axis_vec.get(dload.direction, (0.0, 1.0, 0.0))
                    ox, oy, oz = -sgn * 0.8 * ux, -sgn * 0.8 * uy, -sgn * 0.8 * uz

                    self.ax.plot([xi + ox, xj + ox], [zi + oz, zj + oz], [yi + oy, yj + oy],
                                 color=load_color, linestyle='-', linewidth=1.5)
                    for t in np.linspace(0, 1, 5):
                        px, py, pz = xi + t*(xj - xi), yi + t*(yj - yi), zi + t*(zj - zi)
                        self.ax.quiver(px + ox, pz + oz, py + oy, sgn*ux, sgn*uz, sgn*uy,
                                       color=load_color, length=0.8, normalize=True)

                    mx_, my_, mz_ = (xi + xj)/2 + ox, (yi + yj)/2 + oy, (zi + zj)/2 + oz
                    dist_unit = "kips/ft" if is_imperial else "kN/m"
                    dist_mult = 0.0000685218 if is_imperial else 0.001
                    self.ax.text(mx_, mz_, my_, f" {abs(dload.magnitude)*dist_mult:.2f} {dist_unit}", color=load_color, fontsize=7, weight='bold')

            for pload in load_case.member_point_loads:
                if pload.member_id in mem_map and pload.magnitude != 0:
                    mem = mem_map[pload.member_id]
                    xi, yi, zi = [c * len_mult for c in nodes[mem['ni']]]
                    xj, yj, zj = [c * len_mult for c in nodes[mem['nj']]]
                    r = pload.location_ratio
                    px, py, pz = xi + r*(xj - xi), yi + r*(yj - yi), zi + r*(zj - zi)

                    sgn = 1.0 if pload.magnitude > 0 else -1.0
                    ux, uy, uz = axis_vec.get(pload.direction, (0.0, 1.0, 0.0))
                    tx, ty, tz = px - sgn*1.2*ux, py - sgn*1.2*uy, pz - sgn*1.2*uz

                    self.ax.quiver(tx, tz, ty, sgn*ux, sgn*uz, sgn*uy, color=load_color, length=1.2, normalize=True, linewidth=2)
                    self.ax.text(tx, tz, ty, f" {abs(pload.magnitude)*force_mult:.1f} {force_unit}", color=load_color, fontsize=8, weight='bold')

            if load_case.self_weight_factor > 0:
                d = 1.0 if load_case.sw_factor_dir > 0 else -1.0
                for nid, coords in nodes.items():
                    x, y, z = [c * len_mult for c in coords]
                    self.ax.quiver(x, z, y - d*0.8, 0.0, 0.0, d, color=load_color, length=0.8, normalize=True)

        # Axis Labels
        self.ax.set_xlabel(f"X ({len_unit}) - lateral", color='#681842', fontsize=9, weight='bold')
        self.ax.set_ylabel(f"Z ({len_unit}) - lateral", color='#681842', fontsize=9, weight='bold')
        self.ax.set_zlabel(f"Y ({len_unit}) - vertical", color='#681842', fontsize=9, weight='bold')
        self.ax.tick_params(colors='#8c265c', labelsize=8)

        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#e0b0be')
        self.ax.yaxis.pane.set_edgecolor('#e0b0be')
        self.ax.zaxis.pane.set_edgecolor('#e0b0be')

        lc_text = f"LC{load_case.id}: {load_case.name}" if load_case else "Undeformed"
        self.ax.set_title(f"Structural Model • {lc_text}",
                          color='#520c31', fontsize=11, weight='bold', pad=10)
        self.draw()


# ==============================================================================
# 4. PYQT6 MAIN DASHBOARD WINDOW
# ==============================================================================
class Rev4DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("REV4 — 3D FRAME MATRIX SOLVER")
        self.resize(1560, 920)

        self.dim_x = 6.0
        self.dim_y = 6.0
        self.dim_z = 6.0
        self.unit_system = "Standard Metric"
        self.selected_material = "ASTM A36 Steel"
        self.selected_section = "W150X13.5"
        self.support_type = "Fixed Base"
        self.diaphragm_enabled = True
        self.def_scale = 50.0

        self.materials_db = {
            "ASTM A36 Steel": {'E': 200e9, 'G': 79.3e9, 'Fy': 250e6, 'Poisson': 0.26, 'density': 7850},
            "Grade 50 Steel": {'E': 200e9, 'G': 79.3e9, 'Fy': 345e6, 'Poisson': 0.26, 'density': 7850}
        }
        self.sections_db = {
            "W150X13.5": {'A': 0.00171, 'Iy': 1.66e-6, 'Iz': 5.58e-6, 'J': 2.84e-8, 'd': 150, 'bf': 100},
            "W200X15": {'A': 0.00191, 'Iy': 2.12e-6, 'Iz': 8.51e-6, 'J': 3.50e-8, 'd': 200, 'bf': 100}
        }

        self.solver = None
        self.displacements = None
        self.reactions = None

        self.build_model()
        self.setup_ui()
        self.setup_menu_bar()
        self.update_3d_plot()

    def build_model(self):
        x, y, z = self.dim_x, self.dim_y, self.dim_z

        self.nodes = {
            1: (0.0, 0.0, 0.0), 2: (x, 0.0, 0.0), 3: (x, 0.0, z), 4: (0.0, 0.0, z),
            5: (0.0, y, 0.0), 6: (x, y, 0.0), 7: (x, y, z), 8: (0.0, y, z)
        }

        member_defs = [
            (1, 2, 'Beam'), (2, 3, 'Beam'), (3, 4, 'Beam'), (4, 1, 'Beam'),
            (5, 6, 'Beam'), (6, 7, 'Beam'), (7, 8, 'Beam'), (8, 5, 'Beam'),
            (1, 5, 'Column'), (2, 6, 'Column'), (3, 7, 'Column'), (4, 8, 'Column')
        ]
        self.members = [
            {'id': i + 1, 'ni': ni, 'nj': nj, 'type': mtype,
             'mat': self.selected_material, 'sec': self.selected_section, 'beta': 0.0}
            for i, (ni, nj, mtype) in enumerate(member_defs)
        ]

        self.apply_support_setting()
        self.apply_diaphragm_setting()
        self.build_load_cases()

    def apply_support_setting(self):
        vec = [1, 1, 1, 1, 1, 1] if self.support_type == "Fixed Base" else [1, 1, 1, 0, 0, 0]
        self.supports = {nid: list(vec) for nid in (1, 2, 3, 4)}

    def apply_diaphragm_setting(self):
        if self.diaphragm_enabled:
            self.diaphragms = [
                Diaphragm(1, "Roof Rigid Diaphragm", elevation=self.dim_y, master_node=5, constrained_nodes=[6, 7, 8])
            ]
        else:
            self.diaphragms = []

    def build_load_cases(self):
        self.load_cases = {}

        self.load_cases[1] = LoadCase(1, "DEAD / SELF WEIGHT", "Dead", self_weight_factor=1.0)

        lc2 = LoadCase(2, "ROOF DEAD", "Dead")
        for mem_id in [5, 6, 7, 8]:
            lc2.member_dist_loads.append(MemberDistributedLoad(mem_id, direction='Y', magnitude=-5000.0))
        self.load_cases[2] = lc2

        lc3 = LoadCase(3, "ROOF LIVE", "Live")
        for mem_id in [5, 6, 7, 8]:
            lc3.member_dist_loads.append(MemberDistributedLoad(mem_id, direction='Y', magnitude=-3000.0))
        self.load_cases[3] = lc3

        lc4 = LoadCase(4, "ROOF BEAM CENTER LOAD", "Live")
        for mem_id in [5, 6, 7, 8]:
            lc4.member_point_loads.append(MemberPointLoad(mem_id, location_ratio=0.5, direction='Y', magnitude=-5000.0))
        self.load_cases[4] = lc4

        lc5 = LoadCase(5, "WIND X", "Wind")
        for nid in [5, 6, 7, 8]:
            lc5.nodal_loads.append(NodalLoad(nid, fx=2500.0))
        self.load_cases[5] = lc5

        lc6 = LoadCase(6, "WIND Z", "Wind")
        for nid in [5, 6, 7, 8]:
            lc6.nodal_loads.append(NodalLoad(nid, fz=2500.0))
        self.load_cases[6] = lc6

        lc7 = LoadCase(7, "SEISMIC X", "Seismic")
        for nid in [5, 6, 7, 8]:
            lc7.nodal_loads.append(NodalLoad(nid, fx=3750.0))
        self.load_cases[7] = lc7

        lc8 = LoadCase(8, "SEISMIC Z", "Seismic")
        for nid in [5, 6, 7, 8]:
            lc8.nodal_loads.append(NodalLoad(nid, fz=3750.0))
        self.load_cases[8] = lc8

        lc9 = LoadCase(9, "REV 4 BENCHMARK (10 kN @ N5)", "Benchmark")
        lc9.nodal_loads.append(NodalLoad(5, fx=10000.0))
        self.load_cases[9] = lc9

    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #fff0f5; }
            QMenuBar { background-color: #520c31; color: #ffffff; font-weight: bold; padding: 2px; }
            QMenuBar::item { background-color: transparent; padding: 4px 10px; }
            QMenuBar::item:selected { background-color: #8c1d58; color: #ffffff; border-radius: 3px; }
            QMenu { background-color: #ffffff; color: #520c31; border: 1px solid #f8bbd0; }
            QMenu::item:selected { background-color: #fce4ec; color: #8c1d58; }
            
            QToolBar { background-color: #fce4ec; border-bottom: 1px solid #f8bbd0; padding: 4px; spacing: 6px; }
            QToolBar QPushButton { color: #ffffff; background-color: #d64d8a; border: 1px solid #c23b77; border-radius: 4px; padding: 5px 10px; font-weight: bold; }
            QToolBar QPushButton:hover { background-color: #e05638; border-color: #c53030; color: #ffffff; }
            
            QGroupBox { font-weight: bold; border: 1px solid #f8bbd0; border-radius: 6px; margin-top: 8px; padding-top: 10px; background-color: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #8c1d58; }
            
            QPushButton { background-color: #d64d8a; color: #ffffff; border-radius: 4px; padding: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #c23b77; color: #ffffff; }
            
            QTableWidget, QTreeWidget { gridline-color: #f8bbd0; background-color: #ffffff; selection-background-color: #f48fb1; selection-color: #200b21; border: 1px solid #f8bbd0; border-radius: 4px; color: #333333; }
            QHeaderView::section { background-color: #d64d8a; color: #ffffff; padding: 4px; font-weight: bold; border: none; }
            
            QTabWidget::pane { border: 1px solid #f8bbd0; border-radius: 4px; background-color: #ffffff; }
            QTabBar::tab { background: #fce4ec; color: #8c1d58; padding: 6px 12px; border-top-left-radius: 4px; border-top-right-radius: 4px; font-weight: bold; margin-right: 2px; }
            QTabBar::tab:selected { background: #d64d8a; color: #ffffff; }
            
            QComboBox, QDoubleSpinBox, QLineEdit { background-color: #ffffff; border: 1px solid #f8bbd0; border-radius: 3px; padding: 3px; color: #520c31; font-weight: bold; }
            QComboBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border: 1px solid #d64d8a; }
            
            QTextEdit { border: 1px solid #f8bbd0; background-color: #fff9fb; color: #520c31; font-family: Consolas, monospace; }
            QStatusBar { background-color: #520c31; color: #ffffff; font-weight: bold; }
        """)

        # Main Toolbar
        toolbar = QToolBar("Main Controls", self)
        self.addToolBar(toolbar)

        btn_run = QPushButton("▶ Run Analysis")
        btn_run.clicked.connect(self.run_analysis)
        toolbar.addWidget(btn_run)

        btn_val = QPushButton("✔ Validate Model")
        btn_val.clicked.connect(self.validate_model_dialog)
        toolbar.addWidget(btn_val)

        toolbar.addSeparator()
        btn_reset = QPushButton("↺ Reset Benchmark")
        btn_reset.clicked.connect(self.reset_benchmark)
        toolbar.addWidget(btn_reset)

        toolbar.addSeparator()
        btn_save = QPushButton("💾 Save Model")
        btn_save.clicked.connect(self.save_input_excel)
        toolbar.addWidget(btn_save)

        btn_export = QPushButton("📊 Export Results")
        btn_export.clicked.connect(self.export_results_excel)
        toolbar.addWidget(btn_export)

        # Splitter Layout (3 Panel Architecture)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        # ----------------------------------------------------------------------
        # LEFT PANEL: Properties & Input Controls
        # ----------------------------------------------------------------------
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)

        # Properties Banner Header
        lbl_prop_hdr = QLabel(" Properties")
        lbl_prop_hdr.setStyleSheet("background-color: #8c1d58; color: white; font-weight: bold; padding: 4px; border-radius: 3px;")
        left_layout.addWidget(lbl_prop_hdr)

        # Project Info Box
        proj_box = QGroupBox("Project Information")
        proj_grid = QGridLayout(proj_box)
        proj_grid.addWidget(QLabel("Model Name:"), 0, 0)
        proj_grid.addWidget(QLineEdit("Cube 6 m"), 0, 1)
        proj_grid.addWidget(QLabel("Designer:"), 1, 0)
        proj_grid.addWidget(QLineEdit("Structural Lab"), 1, 1)
        left_layout.addWidget(proj_box)

        # Geometry Box
        geom_box = QGroupBox("Model Geometry")
        geom_grid = QGridLayout(geom_box)

        self.lbl_x = QLabel("X Span (m):")
        self.lbl_y = QLabel("Y Height (m):")
        self.lbl_z = QLabel("Z Depth (m):")

        geom_grid.addWidget(self.lbl_x, 0, 0)
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(1.0, 500.0)
        self.spin_x.setSingleStep(0.5)
        self.spin_x.setValue(self.dim_x)
        geom_grid.addWidget(self.spin_x, 0, 1)

        geom_grid.addWidget(self.lbl_y, 1, 0)
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(1.0, 500.0)
        self.spin_y.setSingleStep(0.5)
        self.spin_y.setValue(self.dim_y)
        geom_grid.addWidget(self.spin_y, 1, 1)

        geom_grid.addWidget(self.lbl_z, 2, 0)
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setRange(1.0, 500.0)
        self.spin_z.setSingleStep(0.5)
        self.spin_z.setValue(self.dim_z)
        geom_grid.addWidget(self.spin_z, 2, 1)

        btn_rebuild = QPushButton("Rebuild Frame Geometry")
        btn_rebuild.clicked.connect(self.rebuild_frame_geometry)
        geom_grid.addWidget(btn_rebuild, 3, 0, 1, 2)

        left_layout.addWidget(geom_box)

        # Section & Material Database Box
        db_box = QGroupBox("Model Input & Libraries")
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

        db_grid.addWidget(QLabel("Base Supports:"), 3, 0)
        self.combo_support = QComboBox()
        self.combo_support.addItems(["Fixed Base", "Pinned Base"])
        self.combo_support.setCurrentText(self.support_type)
        db_grid.addWidget(self.combo_support, 3, 1)

        self.chk_diaphragm = QCheckBox("Rigid Roof Diaphragm")
        self.chk_diaphragm.setChecked(self.diaphragm_enabled)
        db_grid.addWidget(self.chk_diaphragm, 4, 0, 1, 2)

        left_layout.addWidget(db_box)

        # Active Load Case Selector
        lc_box = QGroupBox("Load Case Inspector")
        lc_grid = QGridLayout(lc_box)

        lc_grid.addWidget(QLabel("Active Load Case:"), 0, 0)
        self.combo_lc = QComboBox()
        for lcid in sorted(self.load_cases.keys()):
            self.combo_lc.addItem(f"LC{lcid}: {self.load_cases[lcid].name}", lcid)
        lc_grid.addWidget(self.combo_lc, 0, 1)

        left_layout.addWidget(lc_box)
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left_content)
        splitter.addWidget(left_scroll)

        # ----------------------------------------------------------------------
        # CENTER PANEL: 3D Viewport & Display Controls
        # ----------------------------------------------------------------------
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        # Sub-Toolbar for Visual Toggles
        vis_bar = QHBoxLayout()
        self.chk_nodes = QCheckBox("Nodes")
        self.chk_nodes.setChecked(True)
        self.chk_mems = QCheckBox("Members")
        self.chk_mems.setChecked(True)
        self.chk_deformed = QCheckBox("Deformed Shape")
        self.chk_deformed.setChecked(True)
        self.chk_loads = QCheckBox("Loads")
        self.chk_loads.setChecked(True)
        self.chk_dia_links = QCheckBox("Diaphragm")
        self.chk_dia_links.setChecked(True)

        vis_bar.addWidget(QLabel("<b>View Filters:</b>"))
        vis_bar.addWidget(self.chk_nodes)
        vis_bar.addWidget(self.chk_mems)
        vis_bar.addWidget(self.chk_deformed)
        vis_bar.addWidget(self.chk_loads)
        vis_bar.addWidget(self.chk_dia_links)
        vis_bar.addStretch()

        vis_bar.addWidget(QLabel("Scale:"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(1.0, 1000.0)
        self.spin_scale.setValue(self.def_scale)
        vis_bar.addWidget(self.spin_scale)

        center_layout.addLayout(vis_bar)

        self.canvas = Structure3DCanvas(self)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self)

        self.lbl_info = QLabel("STATUS: Ready • Cube Model Loaded.")
        self.lbl_info.setStyleSheet(
            "font-weight: bold; color: #8c1d58; padding: 5px; background-color: #fce4ec; border: 1px solid #f8bbd0; border-radius: 4px;"
        )

        center_layout.addWidget(self.canvas_toolbar)
        center_layout.addWidget(self.canvas)
        center_layout.addWidget(self.lbl_info)
        splitter.addWidget(center_widget)

        # ----------------------------------------------------------------------
        # RIGHT PANEL: Explorer Tree & Data Tables
        # ----------------------------------------------------------------------
        right_dock = QWidget()
        right_layout = QVBoxLayout(right_dock)

        lbl_exp_hdr = QLabel(" Explorer & Output Data")
        lbl_exp_hdr.setStyleSheet("background-color: #8c1d58; color: white; font-weight: bold; padding: 4px; border-radius: 3px;")
        right_layout.addWidget(lbl_exp_hdr)

        self.tabs = QTabWidget()

        # Data Entry Tree Structure (Mimicking Reference Solver)
        self.tree_explorer = QTreeWidget()
        self.tree_explorer.setHeaderLabel("Model Navigation Tree")
        root_data = QTreeWidgetItem(self.tree_explorer, ["Data Entry"])
        QTreeWidgetItem(root_data, ["Project Grid"])
        QTreeWidgetItem(root_data, ["Materials & Sections"])
        QTreeWidgetItem(root_data, ["Node Coordinates"])
        QTreeWidgetItem(root_data, ["Boundary Conditions"])
        QTreeWidgetItem(root_data, ["Diaphragms"])

        root_loads = QTreeWidgetItem(self.tree_explorer, ["Load Conditions"])
        QTreeWidgetItem(root_loads, ["Basic Load Cases"])
        QTreeWidgetItem(root_loads, ["Nodal Loads"])
        QTreeWidgetItem(root_loads, ["Distributed Loads"])

        root_res = QTreeWidgetItem(self.tree_explorer, ["Analysis Results"])
        QTreeWidgetItem(root_res, ["Nodal Reactions"])
        QTreeWidgetItem(root_res, ["Nodal Deflections"])
        self.tree_explorer.expandAll()

        self.tabs.addTab(self.tree_explorer, "Explorer")

        self.table_nodes = QTableWidget(len(self.nodes), 4)
        self.populate_nodes_table()
        self.tabs.addTab(self.table_nodes, "Nodes")

        self.table_disp = QTableWidget(0, 7)
        self.tabs.addTab(self.table_disp, "Displacements")

        self.table_react = QTableWidget(0, 7)
        self.tabs.addTab(self.table_react, "Reactions")

        self.txt_audit = QTextEdit()
        self.txt_audit.setReadOnly(True)
        self.tabs.addTab(self.txt_audit, "Audit Report")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_dock)

        splitter.setSizes([320, 780, 460])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready | Unit System: Standard Metric • 8 Nodes • 12 Members")

        # Connected Signals
        self.combo_units.currentIndexChanged.connect(self.on_units_changed)
        self.combo_mat.currentIndexChanged.connect(self.on_library_changed)
        self.combo_sec.currentIndexChanged.connect(self.on_library_changed)
        self.combo_support.currentIndexChanged.connect(self.on_support_changed)
        self.chk_diaphragm.stateChanged.connect(self.on_diaphragm_toggled)
        self.combo_lc.currentIndexChanged.connect(self.on_load_case_changed)
        self.chk_nodes.stateChanged.connect(self.update_3d_plot)
        self.chk_mems.stateChanged.connect(self.update_3d_plot)
        self.chk_deformed.stateChanged.connect(self.update_3d_plot)
        self.chk_loads.stateChanged.connect(self.update_3d_plot)
        self.chk_dia_links.stateChanged.connect(self.update_3d_plot)
        self.spin_scale.valueChanged.connect(self.on_scale_change)

        self.validate_and_audit_all_cases()

    def setup_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        new_action = QAction("New Model", self)
        new_action.triggered.connect(self.reset_benchmark)
        file_menu.addAction(new_action)

        save_action = QAction("Save Model Excel...", self)
        save_action.triggered.connect(self.save_input_excel)
        file_menu.addAction(save_action)

        export_action = QAction("Export Results Excel...", self)
        export_action.triggered.connect(self.export_results_excel)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        model_menu = menu_bar.addMenu("&Model")
        val_action = QAction("Validate Model", self)
        val_action.triggered.connect(self.validate_model_dialog)
        model_menu.addAction(val_action)

        analysis_menu = menu_bar.addMenu("&Analysis")
        run_action = QAction("Run Structural Analysis", self)
        run_action.triggered.connect(self.run_analysis)
        analysis_menu.addAction(run_action)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("About Rev4 Solver", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    # --------------------------------------------------------------------------
    # TABLE & STATE HELPERS
    # --------------------------------------------------------------------------
    def populate_nodes_table(self):
        is_imperial = (self.unit_system == "Imperial")
        unit_label = "ft" if is_imperial else "m"
        mult = 3.28084 if is_imperial else 1.0

        self.table_nodes.setHorizontalHeaderLabels(["Node ID", f"X ({unit_label})", f"Y ({unit_label})", f"Z ({unit_label})"])
        self.table_nodes.setRowCount(len(self.nodes))
        for row, (nid, (x, y, z)) in enumerate(self.nodes.items()):
            self.table_nodes.setItem(row, 0, QTableWidgetItem(str(nid)))
            self.table_nodes.setItem(row, 1, QTableWidgetItem(f"{x * mult:.2f}"))
            self.table_nodes.setItem(row, 2, QTableWidgetItem(f"{y * mult:.2f}"))
            self.table_nodes.setItem(row, 3, QTableWidgetItem(f"{z * mult:.2f}"))

    def get_active_load_case(self):
        lc_id = self.combo_lc.currentData()
        return self.load_cases.get(lc_id)

    def invalidate_results(self):
        self.displacements = None
        self.reactions = None
        self.table_disp.setRowCount(0)
        self.table_react.setRowCount(0)
        self.status_bar.showMessage(f"Status: Model Modified — Re-run Analysis required | Units: {self.unit_system}")

    # --------------------------------------------------------------------------
    # CALLBACKS & UNIT CONVERSIONS
    # --------------------------------------------------------------------------
    def on_units_changed(self):
        new_units = self.combo_units.currentText()
        if new_units == self.unit_system:
            return

        self.spin_x.blockSignals(True)
        self.spin_y.blockSignals(True)
        self.spin_z.blockSignals(True)

        if new_units == "Imperial":
            self.lbl_x.setText("X Span (ft):")
            self.lbl_y.setText("Y Height (ft):")
            self.lbl_z.setText("Z Depth (ft):")
            self.spin_x.setValue(self.dim_x * 3.28084)
            self.spin_y.setValue(self.dim_y * 3.28084)
            self.spin_z.setValue(self.dim_z * 3.28084)
        else:
            self.lbl_x.setText("X Span (m):")
            self.lbl_y.setText("Y Height (m):")
            self.lbl_z.setText("Z Depth (m):")
            self.spin_x.setValue(self.dim_x)
            self.spin_y.setValue(self.dim_y)
            self.spin_z.setValue(self.dim_z)

        self.spin_x.blockSignals(False)
        self.spin_y.blockSignals(False)
        self.spin_z.blockSignals(False)

        self.unit_system = new_units
        self.populate_nodes_table()
        self.invalidate_results()
        self.update_3d_plot()

    def rebuild_frame_geometry(self):
        val_x = self.spin_x.value()
        val_y = self.spin_y.value()
        val_z = self.spin_z.value()

        if self.unit_system == "Imperial":
            self.dim_x = val_x / 3.28084
            self.dim_y = val_y / 3.28084
            self.dim_z = val_z / 3.28084
        else:
            self.dim_x = val_x
            self.dim_y = val_y
            self.dim_z = val_z

        self.selected_material = self.combo_mat.currentText()
        self.selected_section = self.combo_sec.currentText()
        self.support_type = self.combo_support.currentText()
        self.diaphragm_enabled = self.chk_diaphragm.isChecked()

        self.build_model()
        self.populate_nodes_table()
        self.invalidate_results()
        self.validate_and_audit_all_cases()
        self.update_3d_plot()

        unit_str = "ft" if self.unit_system == "Imperial" else "m"
        QMessageBox.information(
            self,
            "Geometry Updated",
            f"Frame rebuilt successfully to dimensions: {val_x:.1f}{unit_str} x {val_y:.1f}{unit_str} x {val_z:.1f}{unit_str}."
        )

    def on_scale_change(self):
        self.def_scale = self.spin_scale.value()
        self.update_3d_plot()

    def on_library_changed(self):
        self.selected_material = self.combo_mat.currentText()
        self.selected_section = self.combo_sec.currentText()
        for mem in self.members:
            mem['mat'] = self.selected_material
            mem['sec'] = self.selected_section
        self.invalidate_results()
        self.validate_and_audit_all_cases()
        self.update_3d_plot()

    def on_support_changed(self):
        self.support_type = self.combo_support.currentText()
        self.apply_support_setting()
        self.invalidate_results()
        self.update_3d_plot()

    def on_diaphragm_toggled(self):
        self.diaphragm_enabled = self.chk_diaphragm.isChecked()
        self.apply_diaphragm_setting()
        self.invalidate_results()
        self.update_3d_plot()

    def on_load_case_changed(self):
        self.invalidate_results()
        self.update_3d_plot()

    def update_3d_plot(self):
        self.canvas.plot_structure(
            self.nodes, self.members, self.displacements, scale=self.def_scale,
            show_node_labels=self.chk_nodes.isChecked(),
            show_mem_labels=self.chk_mems.isChecked(),
            show_loads=self.chk_loads.isChecked(),
            show_deformed=self.chk_deformed.isChecked(),
            show_diaphragms=self.chk_dia_links.isChecked(),
            load_case=self.get_active_load_case(),
            supports=self.supports, diaphragms=self.diaphragms,
            unit_system=self.unit_system
        )

    # --------------------------------------------------------------------------
    # VALIDATION & AUDIT
    # --------------------------------------------------------------------------
    def validate_and_audit_all_cases(self):
        report = "==================================================\n"
        report += "      REV4 MODEL VALIDATION & LOAD AUDIT REPORT\n"
        report += "==================================================\n\n"
        report += f"Unit System   : {self.unit_system}\n"
        report += f"Base Supports : {self.support_type}\n"
        report += f"Roof Diaphragm: {'Enabled' if self.diaphragm_enabled else 'Disabled'}\n\n"

        solver = DirectStiffness3DSolver(
            self.nodes, self.members, self.supports, {}, self.materials_db, self.sections_db,
            self.unit_system, diaphragms=self.diaphragms
        )

        mem_map = {m['id']: m for m in self.members}
        dir_idx = {'X': 0, 'Y': 1, 'Z': 2}
        is_imp = (self.unit_system == "Imperial")
        f_mult = 0.000224809 if is_imp else 0.001
        f_str = "kips" if is_imp else "kN"

        for lcid in sorted(self.load_cases.keys()):
            lc = self.load_cases[lcid]
            F_glob = solver.build_load_vector(lc)
            fx_tot, fy_tot, fz_tot = np.sum(F_glob[0::6]), np.sum(F_glob[1::6]), np.sum(F_glob[2::6])

            expected = np.zeros(3)
            for nl in lc.nodal_loads:
                expected += [nl.fx, nl.fy, nl.fz]
            for dl in lc.member_dist_loads:
                if dl.member_id in mem_map:
                    m = mem_map[dl.member_id]
                    L = np.linalg.norm(np.array(self.nodes[m['nj']]) - np.array(self.nodes[m['ni']]))
                    expected[dir_idx.get(dl.direction, 1)] += dl.magnitude * L
            for pl in lc.member_point_loads:
                if pl.member_id in mem_map:
                    expected[dir_idx.get(pl.direction, 1)] += pl.magnitude
            if lc.self_weight_factor > 0:
                for m in self.members:
                    L = np.linalg.norm(np.array(self.nodes[m['nj']]) - np.array(self.nodes[m['ni']]))
                    dens = self.materials_db.get(m['mat'], {}).get('density', 7850)
                    area = self.sections_db.get(m['sec'], {}).get('A', 0.00171)
                    expected[1] += lc.sw_factor_dir * dens * area * L * 9.81 * lc.self_weight_factor

            ok = np.allclose([fx_tot, fy_tot, fz_tot], expected, rtol=1e-6, atol=1e-6)

            report += f"Load Case {lcid}: {lc.name}\n"
            report += f"  - Category: {lc.category}\n"
            report += f"  - Total FX: {fx_tot*f_mult:.3f} {f_str} | Total FY: {fy_tot*f_mult:.3f} {f_str} | Total FZ: {fz_tot*f_mult:.3f} {f_str}\n"
            report += f"  - Status: {'VALIDATED' if ok else 'MISMATCH'}\n"
            report += "--------------------------------------------------\n"

        self.txt_audit.setText(report)

    def validate_model_dialog(self):
        self.validate_and_audit_all_cases()
        self.tabs.setCurrentWidget(self.txt_audit)
        QMessageBox.information(self, "Validation", "Model validation completed without errors. Details shown in Audit tab.")

    # --------------------------------------------------------------------------
    # ANALYSIS & RUNTIME
    # --------------------------------------------------------------------------
    def run_analysis(self):
        try:
            lc = self.get_active_load_case()
            if lc is None:
                raise ValueError("No load case selected.")

            self.solver = DirectStiffness3DSolver(
                self.nodes, self.members, self.supports, {},
                self.materials_db, self.sections_db, self.unit_system,
                diaphragms=self.diaphragms
            )
            K_glob = self.solver.assemble_global_stiffness()
            F_glob = self.solver.build_load_vector(lc)
            self.displacements, self.reactions = self.solver.solve_load_case(K_glob, F_glob)

            node_id_map = {nid: idx for idx, nid in enumerate(sorted(self.nodes.keys()))}

            is_imp = (self.unit_system == "Imperial")
            disp_unit = "in" if is_imp else "mm"
            disp_mult = 39.3701 if is_imp else 1000.0

            force_unit = "kips" if is_imp else "kN"
            force_mult = 0.000224809 if is_imp else 0.001

            mom_unit = "kip-ft" if is_imp else "kNm"
            mom_mult = 0.000737562 if is_imp else 0.001

            self.table_disp.setHorizontalHeaderLabels([
                "Node", f"UX ({disp_unit})", f"UY ({disp_unit})", f"UZ ({disp_unit})",
                "RX (rad)", "RY (rad)", "RZ (rad)"
            ])
            self.table_disp.setRowCount(len(self.nodes))
            max_disp = 0.0
            max_node = 1

            for row, nid in enumerate(sorted(self.nodes.keys())):
                idx = node_id_map[nid]
                d = self.displacements[idx*6 : (idx+1)*6]
                trans_mag = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2) * disp_mult
                if trans_mag > max_disp:
                    max_disp = trans_mag
                    max_node = nid

                self.table_disp.setItem(row, 0, QTableWidgetItem(f"Node {nid}"))
                self.table_disp.setItem(row, 1, QTableWidgetItem(f"{d[0]*disp_mult:.4f}"))
                self.table_disp.setItem(row, 2, QTableWidgetItem(f"{d[1]*disp_mult:.4f}"))
                self.table_disp.setItem(row, 3, QTableWidgetItem(f"{d[2]*disp_mult:.4f}"))
                self.table_disp.setItem(row, 4, QTableWidgetItem(f"{d[3]:.6f}"))
                self.table_disp.setItem(row, 5, QTableWidgetItem(f"{d[4]:.6f}"))
                self.table_disp.setItem(row, 6, QTableWidgetItem(f"{d[5]:.6f}"))

            self.table_react.setHorizontalHeaderLabels([
                "Node", f"FX ({force_unit})", f"FY ({force_unit})", f"FZ ({force_unit})",
                f"MX ({mom_unit})", f"MY ({mom_unit})", f"MZ ({mom_unit})"
            ])
            self.table_react.setRowCount(len(self.supports))
            for row, nid in enumerate(sorted(self.supports.keys())):
                idx = node_id_map[nid]
                r = self.reactions[idx*6 : (idx+1)*6]
                self.table_react.setItem(row, 0, QTableWidgetItem(f"Node {nid}"))
                self.table_react.setItem(row, 1, QTableWidgetItem(f"{r[0]*force_mult:.2f}"))
                self.table_react.setItem(row, 2, QTableWidgetItem(f"{r[1]*force_mult:.2f}"))
                self.table_react.setItem(row, 3, QTableWidgetItem(f"{r[2]*force_mult:.2f}"))
                self.table_react.setItem(row, 4, QTableWidgetItem(f"{r[3]*mom_mult:.2f}"))
                self.table_react.setItem(row, 5, QTableWidgetItem(f"{r[4]*mom_mult:.2f}"))
                self.table_react.setItem(row, 6, QTableWidgetItem(f"{r[5]*mom_mult:.2f}"))

            status_msg = f"Analysis Complete | Max Disp: {max_disp:.3f} {disp_unit} at Node {max_node}"
            self.lbl_info.setText(f"STATUS: {status_msg}")
            self.status_bar.showMessage(f"Status: Complete | Active LC: {lc.name} | Units: {self.unit_system}")
            self.tabs.setCurrentWidget(self.table_disp)
            self.update_3d_plot()
            QMessageBox.information(self, "Analysis Complete", f"Direct Stiffness Method solved successfully for {lc.name}.")

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to solve structure:\n{str(e)}")

    def reset_benchmark(self):
        controls = [self.spin_x, self.spin_y, self.spin_z, self.combo_units, self.combo_mat,
                    self.combo_sec, self.combo_support, self.chk_diaphragm, self.combo_lc]
        for w in controls:
            w.blockSignals(True)

        self.spin_x.setValue(6.0)
        self.spin_y.setValue(6.0)
        self.spin_z.setValue(6.0)
        self.combo_units.setCurrentIndex(0)
        self.combo_mat.setCurrentIndex(0)
        self.combo_sec.setCurrentIndex(0)
        self.combo_support.setCurrentText("Pinned Base")
        self.chk_diaphragm.setChecked(False)
        self.combo_lc.setCurrentIndex(self.combo_lc.findData(9))

        for w in controls:
            w.blockSignals(False)

        self.dim_x, self.dim_y, self.dim_z = 6.0, 6.0, 6.0
        self.unit_system = "Standard Metric"
        self.selected_material = self.combo_mat.currentText()
        self.selected_section = self.combo_sec.currentText()
        self.support_type = "Pinned Base"
        self.diaphragm_enabled = False

        self.lbl_x.setText("X Span (m):")
        self.lbl_y.setText("Y Height (m):")
        self.lbl_z.setText("Z Depth (m):")

        self.build_model()
        self.invalidate_results()
        self.populate_nodes_table()
        self.validate_and_audit_all_cases()
        self.update_3d_plot()
        self.lbl_info.setText("STATUS: Reset to REV4 Benchmark Model (6m Cube, Pinned Base, 10 kN @ N5).")

    # --------------------------------------------------------------------------
    # EXCEL I/O & DIALOGS
    # --------------------------------------------------------------------------
    def save_input_excel(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Structural Model", "Structural_Model_Rev4.xlsx", "Excel Files (*.xlsx)")
        if filepath:
            with pd.ExcelWriter(filepath) as writer:
                pd.DataFrame([{
                    'System': self.combo_units.currentText(),
                    'Material': self.combo_mat.currentText(),
                    'Section': self.combo_sec.currentText(),
                    'Supports': self.support_type,
                    'Diaphragm': 'Enabled' if self.diaphragm_enabled else 'Disabled'
                }]).to_excel(writer, sheet_name='Settings', index=False)
                pd.DataFrame([{'Node': k, 'X': v[0], 'Y': v[1], 'Z': v[2]} for k, v in self.nodes.items()]).to_excel(writer, sheet_name='Nodes', index=False)
                pd.DataFrame(self.members).to_excel(writer, sheet_name='Members', index=False)
            QMessageBox.information(self, "Saved", f"Model saved successfully to:\n{filepath}")

    def export_results_excel(self):
        if self.displacements is None:
            QMessageBox.warning(self, "No Results", "Please run analysis before exporting results.")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Export Results", "Structural_Results_Rev4.xlsx", "Excel Files (*.xlsx)")
        if filepath:
            with pd.ExcelWriter(filepath) as writer:
                lc = self.get_active_load_case()
                pd.DataFrame([{
                    'LoadCase': f"LC{lc.id}: {lc.name}" if lc else '',
                    'UnitSystem': self.unit_system,
                    'Supports': self.support_type,
                    'Material': self.combo_mat.currentText(),
                    'Section': self.combo_sec.currentText()
                }]).to_excel(writer, sheet_name='Info', index=False)

                disp_data = []
                node_id_map = {nid: idx for idx, nid in enumerate(sorted(self.nodes.keys()))}
                is_imp = (self.unit_system == "Imperial")
                disp_mult = 39.3701 if is_imp else 1000.0

                for nid in sorted(self.nodes.keys()):
                    idx = node_id_map[nid]
                    d = self.displacements[idx*6 : (idx+1)*6]
                    disp_data.append({
                        'Node': nid,
                        'UX': d[0]*disp_mult,
                        'UY': d[1]*disp_mult,
                        'UZ': d[2]*disp_mult,
                        'RX_rad': d[3],
                        'RY_rad': d[4],
                        'RZ_rad': d[5]
                    })
                pd.DataFrame(disp_data).to_excel(writer, sheet_name='Displacements', index=False)
            QMessageBox.information(self, "Exported", f"Results exported to:\n{filepath}")

    def show_about_dialog(self):
        QMessageBox.about(
            self,
            "About REV4 Solver",
            "<b>REV4 — 3D Frame Matrix Solver</b><br>"
            "Built with PyQt6 & Matplotlib.<br>"
            "Implements full 3D Direct Stiffness Analysis (6 DOF/node) with Master-Slave Diaphragm Constraints."
        )


# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Rev4DashboardWindow()
    window.show()
    sys.exit(app.exec())