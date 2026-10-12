import os
import time
import json
import zlib
import base64
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, request, jsonify
from flask_cors import CORS
from tapipy.tapis import Tapis

app = Flask(__name__)
# Allow CORS for your GitHub Pages domain (update this to your actual GitHub IO URL)
CORS(app, origins=["https://omguptaup.github.io"]) # Change "*" to "https://yourusername.github.io" for production

# Retrieve secrets from Render Environment Variables
TAPIS_USERNAME = os.environ.get("TAPIS_USERNAME")
TAPIS_PASSWORD = os.environ.get("TAPIS_PASSWORD")
ACTOR_ID = os.environ.get("ACTOR_ID")

def cartesian_to_lonlat(x, y, z):
    """Convert Cartesian coordinates on a unit sphere to longitude and latitude in degrees."""
    rad = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / rad)
    lon = np.arctan2(y, x)
    return np.degrees(lon), np.degrees(lat)

def trajectory(Omega, P0):
    """Generates the rotation track around Omega using Rodrigues' rotation formula"""
    Omega_norm = Omega / np.linalg.norm(Omega)
    P0_norm = P0 / np.linalg.norm(P0)

    thetas = np.linspace(0, 2 * np.pi, 200)

    # Rodrigues' rotation formula:
    # v_rot = v * cos(th) + (k x v) * sin(th) + k * (k . v) * (1 - cos(th))
    k = Omega_norm
    v = P0_norm
    k_cross_v = np.cross(k, v)
    k_dot_v = np.dot(k, v)

    cos_t = np.cos(thetas)[:, np.newaxis]
    sin_t = np.sin(thetas)[:, np.newaxis]

    P_traj = v * cos_t + k_cross_v * sin_t + k * k_dot_v * (1.0 - cos_t)

    lon_traj, lat_traj = cartesian_to_lonlat(P_traj[:, 0], P_traj[:, 1], P_traj[:, 2])

    diffs = np.abs(np.diff(lon_traj))
    split_indices = np.where(diffs > 180)[0] + 1
    lon_traj = np.insert(lon_traj, split_indices, np.nan)
    lat_traj = np.insert(lat_traj, split_indices, np.nan)
    return lon_traj, lat_traj

@app.route("/", methods=["GET"])
def health_check():
  return "Backend server is running and healthy!", 200

@app.route("/run-simulation", methods=["POST"])
def run_simulation():
    data = request.json
    choice = int(data.get("choice", 2))
    
    if not all([TAPIS_USERNAME, TAPIS_PASSWORD, ACTOR_ID]):
        return jsonify({"status": "error", "message": "Server credentials not configured."}), 500

    try:
        # 1. Authenticate and Submit to Abaco
        t = Tapis(base_url="https://tacc.tapis.io", username=TAPIS_USERNAME, password=TAPIS_PASSWORD)
        t.get_tokens()
        
        resp = t.actors.send_message(actor_id=ACTOR_ID, message=data)
        exec_id = resp.execution_id

        # 2. Poll for Completion
        while True:
            exec_status = t.actors.get_execution(actor_id=ACTOR_ID, execution_id=exec_id)
            if exec_status.status in ["COMPLETE", "ERROR"]:
                break
            time.sleep(2)

        if exec_status.status == "ERROR":
            return jsonify({"status": "error", "message": "Actor execution failed on TACC."}), 500

        logs = t.actors.get_execution_logs(actor_id=ACTOR_ID, execution_id=exec_id).logs
        
        match = re.search(r'---PAYLOAD_START---\n(.*?)\n---PAYLOAD_END---', logs, re.DOTALL)
        if not match:
            return jsonify({"status": "error", "message": "No valid payload found. Container may have crashed."}), 500
            
        compressed_data = match.group(1).strip()
        json_str = zlib.decompress(base64.b64decode(compressed_data)).decode('utf-8')
        raw_data = json.loads(json_str)

        nu = np.array(raw_data["nu"])
        r = np.array(raw_data["r"])
        stokes_mu = np.array(raw_data["stokes_mu"])
        eta1 = np.array(raw_data["eta1"])
        eta2 = np.array(raw_data["eta2"])
        warning_msg = raw_data.get("warning", "")

        q = stokes_mu[:, :, 0]
        u = stokes_mu[:, :, 1]
        v = stokes_mu[:, :, 2]
        psi = 0.5 * np.arctan2(u, q)
        chi = 0.5 * np.arctan2(v, np.sqrt(q**2 + u**2))

        # Because the container already filtered the frames, we iterate over the exact length
        frames = np.arange(len(r))

        fig = make_subplots(
            rows=1, cols=2, 
            specs=[[{"type": "scattergeo"}, {"type": "xy"}]],
            horizontal_spacing=0.04,
        )

        marker_style_with_cbar = dict(
            color=nu/1e6, colorscale='plasma', size=6, opacity=0.8, showscale=True,
            colorbar=dict(title="Frequency (MHz)", orientation="h", x=0.2, y=-0.15, len=0.4, thickness=15)
        )
        marker_style_no_cbar = dict(color=nu, colorscale='plasma', size=10, opacity=0.8, showscale=False, symbol='x')

        for f in frames:
            fig.add_trace(go.Scattergeo(visible=False, lon=np.degrees((2 * psi[f] + np.pi) % (2 * np.pi) - np.pi), lat=np.degrees(2*chi[f]), 
                          mode='markers', marker=marker_style_with_cbar,name=r"$\mathbf{P}(\nu)$"), row=1, col=1)
            fig.add_trace(go.Scattergeo(visible=False, lon=np.degrees([0]*nu.shape[0]), lat=np.degrees(np.arctan2(eta1[f], eta2[f])),
                          mode='markers', marker=marker_style_no_cbar,name=r"$\mathbf{\Omega}$"), row=1, col=1)
            
            lon_traj, lat_traj = trajectory(np.array([-eta2[f,0], 0, eta1[f,0]]), stokes_mu[f,0,:])
            fig.add_trace(go.Scattergeo(visible=False, lon=lon_traj, lat=lat_traj, mode='lines', line=dict(width=0.5, color='royalblue'), name=f'Trace @ {nu[0]/1e6:.1f} MHz'), row=1, col=1)
            
            if choice == 1:
                fig.add_trace(go.Scatter(visible=False, x=nu/1e6, y=stokes_mu[f,:,2], name=f'V', line=dict(color='black')), row=1, col=2)
                fig.add_trace(go.Scatter(visible=False, x=nu/1e6, y=stokes_mu[f,:,1], name=f'U'), row=1, col=2)
                fig.add_trace(go.Scatter(visible=False, x=nu/1e6, y=stokes_mu[f,:,0], name=f'Q'), row=1, col=2)
            else:
                fig.add_trace(go.Scatter(visible=False, x=nu/1e6, y=psi[f], name=r'$\psi$'), row=1, col=2)
                fig.add_trace(go.Scatter(visible=False, x=nu/1e6, y=chi[f], name=r"$\chi$", line=dict(color='black')), row=1, col=2)

        fig.data[0].visible = True
        fig.data[1].visible = True
        fig.data[2].visible = True
        fig.data[3].visible = True
        fig.data[4].visible = True
        if choice == 1:
            fig.data[5].visible = True

        steps = []
        total_traces = len(fig.data)

        for i in range(len(frames)):
            visibility_mask = [False] * total_traces
            numf = 6 if choice == 1 else 5
            for j in range(numf):
                visibility_mask[numf * i + j] = True
            
            step = dict(
                method="update",
                args=[
                    {"visible": visibility_mask}, 
                    {"title.text": f"Radius: {r[i]*1e13:.3e} cm"} 
                ],
                label=str(i)
            )
            steps.append(step)

        sliders = [dict(active=0, currentvalue={"prefix": "Frame: "}, pad={"t": 50}, steps=steps)]

        fig.update_geos(projection_type="aitoff", showland=False, showocean=False, showcoastlines=False, 
                        showframe=True, framecolor="black", lataxis_showgrid=True, lonaxis_showgrid=True,  
                        lataxis_gridcolor="lightgray", lonaxis_gridcolor="lightgray", lataxis_gridwidth=1, 
                        lonaxis_gridwidth=1, lataxis_dtick=30, lonaxis_dtick=30)
        
        if choice == 1:
            fig.update_yaxes(range=[-1., 1.], row=1, col=2)
        else:
            fig.update_yaxes(range=[-np.pi/2, np.pi/2], row=1, col=2)
            
        fig.update_xaxes(title_text="Frequency (MHz)", row=1, col=2)
        fig.update_layout(sliders=sliders, height=400, width=1000, margin=dict(l=20, r=5, t=60, b=5), title=f"Radius: {r[0]*1e13:.3e} cm")

        # Parse Plotly JSON and send it back to the React/HTML frontend
        plot_json = json.loads(fig.to_json())
        return jsonify({"status": "success", "warning": warning_msg, "plotData": plot_json})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
