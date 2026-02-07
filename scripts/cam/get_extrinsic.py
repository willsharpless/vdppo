import numpy as np
import argparse
import cv2

def estimate_camera_pose(
    object_points_xyz,
    image_points_uv,
    camera_matrix,
    dist_coeffs=None,
    use_ransac=True,
    refine=True,
):
    """
    Estimate camera pose from 3D-2D correspondences.

    Parameters
    ----------
    object_points_xyz : (N,3) array-like
        3D points in WORLD coordinates (e.g., meters).
    image_points_uv : (N,2) array-like
        Corresponding 2D pixel points (u,v).
    camera_matrix : (3,3) array-like
        Intrinsic matrix K from cv2.calibrateCamera.
    dist_coeffs : (k,) array-like or None
        Distortion coefficients from cv2.calibrateCamera (e.g., (5,) or (8,)).
        If you used a distortion model in calibration, pass the same here.
        If you already undistorted points, set dist_coeffs=None or zeros.
    use_ransac : bool
        If True, uses solvePnPRansac to reject outliers.
    refine : bool
        If True, refines pose with solvePnPRefineLM (requires initial pose).

    Returns
    -------
    results : dict with keys:
        rvec, tvec, R, camera_position_world, reprojection_rmse, inliers
    """

    objp = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    imgp = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)

    if objp.shape[0] < 4:
        raise ValueError("Need at least 4 correspondences for PnP.")

    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    if dist_coeffs is None:
        dist = np.zeros((5, 1), dtype=np.float64)  # safe default
    else:
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    # OpenCV expects shapes (N,1,3) and (N,1,2) for some functions
    objp_cv = objp.reshape(-1, 1, 3)
    imgp_cv = imgp.reshape(-1, 1, 2)

    # Choose a PnP method. For many points, ITERATIVE is a solid default.
    pnp_flag = cv2.SOLVEPNP_ITERATIVE

    inliers = None
    if use_ransac:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=objp_cv,
            imagePoints=imgp_cv,
            cameraMatrix=K,
            distCoeffs=dist,
            flags=pnp_flag,
            reprojectionError=3.0,   # pixels; adjust if needed
            confidence=0.999,
            iterationsCount=2000,
        )
        if not ok:
            raise RuntimeError("solvePnPRansac failed to find a valid pose.")
        # Optionally re-run solvePnP on inliers only for a cleaner estimate
        objp_in = objp_cv[inliers[:, 0]]
        imgp_in = imgp_cv[inliers[:, 0]]
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=objp_in,
            imagePoints=imgp_in,
            cameraMatrix=K,
            distCoeffs=dist,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=pnp_flag,
        )
        if not ok:
            raise RuntimeError("solvePnP refinement-on-inliers failed.")
        objp_used, imgp_used = objp_in, imgp_in
    else:
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=objp_cv,
            imagePoints=imgp_cv,
            cameraMatrix=K,
            distCoeffs=dist,
            flags=pnp_flag,
        )
        if not ok:
            raise RuntimeError("solvePnP failed to find a valid pose.")
        objp_used, imgp_used = objp_cv, imgp_cv

    if refine:
        # LM refinement (needs OpenCV contrib in some builds, but usually available)
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=objp_used,
            imagePoints=imgp_used,
            cameraMatrix=K,
            distCoeffs=dist,
            rvec=rvec,
            tvec=tvec,
        )

    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)  # world->camera rotation

    # Camera center in world coordinates: C = -R^T * t
    camera_position_world = (-R.T @ tvec).reshape(3)

    # Reprojection error (RMSE in pixels) on the points used
    proj, _ = cv2.projectPoints(objp_used, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    obs = imgp_used.reshape(-1, 2)
    err = proj - obs
    rmse = np.sqrt(np.mean(np.sum(err**2, axis=1)))

    return {
        "rvec": rvec.reshape(3),
        "tvec": tvec.reshape(3),
        "R": R,
        "camera_position_world": camera_position_world,
        "reprojection_rmse_px": float(rmse),
        "inliers": None if inliers is None else inliers.reshape(-1),
    }

def cube_vertices_unit():
    """8 vertices of cube spanning [0,1]^3 in world coordinates."""
    return np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ],
        dtype=np.float64,
    )


def draw_wireframe_cube(frame, img_pts_2d, thickness=2):
    """
    Draws cube edges on frame.
    img_pts_2d: (8,2) projected pixel points corresponding to cube vertices order above.
    """
    pts = img_pts_2d.astype(int)

    edges = [
        # bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for a, b in edges:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), thickness, cv2.LINE_AA)

    # Optional: draw vertex dots
    for p in pts:
        cv2.circle(frame, tuple(p), 4, (0, 0, 255), -1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Estimate pose from first video frame and draw a [0,1]^3 cube.")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--save", default="", help="Optional path to save the annotated first frame (e.g., out.png).")
    args = parser.parse_args()

    # -----------------------------
    # EXAMPLE INPUTS (replace these)
    # -----------------------------
    # 3D points in your world coordinate system (units: e.g., meters)
    # object_points_xyz = np.array([
    #     # [X, Y, Z],
    #     [9.80838153e-01, 3.68931848e-02, 7.39830678e-04],
    #     [1.25716650e-01, 9.31765074e-02, 4.49284299e-01],
    #     [1.47526643e-01, -1.19057706e+00, 4.54581339e-01],
    #     [-1.27875157e+00, -5.97826832e-01, 7.07832686e-01],
    #     [2.94760639e-02, 1.14852178e+00, 3.01009726e-01],
    #     [2.00494905e+00, -1.11066724e-01, 8.11119967e-01],
    #     [2.03007951e+00, 6.18168001e-01, 3.94497278e-01],
    #     [9.19394279e-01, 7.04895343e-01, 2.66364181e-01],
    #     [2.19976483e-01, 1.85487974e-01, 1.40510192e+00],
    #     [1.37107465e+00, -7.17110435e-01, 1.30108072e+00],
    #     [-6.30467013e-01, -3.14802147e-01, 1.26801314e+00],
    #     [-9.27882355e-01, 5.78938958e-01, 1.24240347e+00],
    #     [1.50218049e+00, -7.94337463e-01, 1.16981903e+00],
    #     [1.53807133e+00, 4.65280013e-01, 4.68960707e-04],
    #     [-4.70989020e-01, 1.12712209e+00, 1.04096406e-03]
    # ], dtype=np.float64)

    object_points_xyz = np.array([[ 2.08360554e+00, -4.66548404e-01,  3.72142442e-01],
       [ 1.20626670e+00, -5.43370759e-01,  4.56488050e-01],
       [ 1.12885814e+00,  8.94201760e-01,  4.68976793e-01],
       [-2.76005021e-01,  8.14334587e-01,  5.08468755e-01],
       [-3.20143458e-01, -7.63216662e-01,  3.18407977e-01],
       [ 4.46508216e-01, -9.77658208e-01, -4.29031096e-03],
       [ 8.61499245e-01,  2.78339546e-01, -2.02249713e-03],
       [ 5.11427258e-01,  1.14753010e+00, -6.11473227e-04],
       [-5.02141339e-01,  1.07230321e+00,  5.13136610e-01],
       [-1.10260110e+00,  1.55033270e-01, -3.68831053e-05],
       [ 5.53266658e-01, -1.59098049e-01, -2.46909999e-03],
       [ 9.25441128e-02, -9.29465167e-01,  1.32362858e+00],
       [ 2.04016443e+00,  8.86667542e-01,  4.35490042e-01],
       [ 2.05262778e+00,  1.40612279e+00,  4.36072252e-01],
       [ 1.72349385e+00,  9.83268193e-01, -7.84826749e-04],
       [ 1.03018631e+00,  4.96512486e-01,  1.14821508e+00],
       [-1.00264832e+00,  5.23370928e-01,  5.95655472e-01],
       [ 7.94417436e-01, -9.72593439e-02,  1.42633294e+00],
       [ 5.87026053e-01, -9.57811503e-01,  1.14258685e+00],
       [ 1.98471413e+00, -1.03301972e+00,  3.41937569e-01]])

    # Matching 2D pixel coordinates (u, v)
    # image_points_uv = np.array([
    #     [1071, 546],
    #     [762, 488],
    #     [735, 1011],
    #     [100, 747],
    #     [767, 158],
    #     [1596, 565],
    #     [1474, 322],
    #     [1061, 305],
    #     [662, 294],
    #     [1403, 931],
    #     [173, 613],
    #     [153, 122],
    #     [1468, 962],
    #     [1251, 420],
    #     [643, 216]
    # ], dtype=np.float64)

    image_points_uv = np.array([
        [1462, 724],
        [1141, 751],
        [1105, 215],
        [584, 218],
        [540, 816],
        [844, 884],
        [993, 472],
        [893, 215],
        [507, 131],
        [370, 493],
        [893, 604],
        [482, 1062],
        [1427, 227],
        [1420, 60],
        [1258, 267],
        [1112, 220],
        [263, 293],
        [994, 475],
        [856, 1030],
        [1437, 946]
    ])

    # Intrinsics from cv2.calibrateCamera
    fx = 900.519466473387
    cx = 952.5860218910146
    fy = 899.5206804119418
    cy = 529.1556457783807
    camera_matrix = np.array([
        [fx,    0.0,  cx],
        [   0.0, fy,  cy],
        [   0.0,    0.0,    1.0]
    ], dtype=np.float64)

    # Distortion from cv2.calibrateCamera (k1,k2,p1,p2,k3) or whatever you estimated
    k1 = -0.0015148732848438218
    k2 = -0.005083101504894888
    p1 = -0.0004602667266030599
    p2 = -0.00394098532185071
    k3 = 0.002377556100158611
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    # -----------------------------
    # ESTIMATE POSE
    # -----------------------------
    results = estimate_camera_pose(
        object_points_xyz=object_points_xyz,
        image_points_uv=image_points_uv,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        use_ransac=True,
        refine=True,
    )

    print("Reprojection RMSE (px):", results["reprojection_rmse_px"])
    if results["inliers"] is not None:
        print("Inliers used:", len(results["inliers"]), "/", len(object_points_xyz))

    print("\nRotation (rvec):", results["rvec"])
    print("Translation (tvec):", results["tvec"])
    print("\nCamera position in WORLD coords (X,Y,Z):", results["camera_position_world"])
    print("\nRotation matrix R (world->camera):\n", repr(results["R"]))
    print()
    print("K matrix:\n", camera_matrix)
    print("Distortion coeffs: ", dist_coeffs)

    # -----------------------------
    rvec = results["rvec"].reshape(3, 1)
    tvec = results["tvec"].reshape(3, 1)
    rmse_px = results["reprojection_rmse_px"]

    # -----------------------------
    # LOAD FIRST FRAME
    # -----------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read the first frame from the video.")

    # -----------------------------
    # PROJECT AND DRAW UNIT CUBE [0,1]^3
    # -----------------------------
    cube_3d = cube_vertices_unit().reshape(-1, 1, 3)
    dist = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    cube_2d, _ = cv2.projectPoints(cube_3d, rvec, tvec, camera_matrix, dist)
    cube_2d = cube_2d.reshape(-1, 2)

    annotated = frame.copy()
    annotated = draw_wireframe_cube(annotated, cube_2d, thickness=2)

    # Optional: show pose text
    R, _ = cv2.Rodrigues(rvec)
    cam_pos_world = (-R.T @ tvec).reshape(3)
    cv2.putText(
        annotated,
        f"RMSE: {rmse_px:.2f}px  CamPos: [{cam_pos_world[0]:.3f}, {cam_pos_world[1]:.3f}, {cam_pos_world[2]:.3f}]",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # -----------------------------
    # DISPLAY / SAVE
    # -----------------------------
    cv2.imshow("Pose + Unit Cube", annotated)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.save:
        cv2.imwrite(args.save, annotated)
        print("Saved:", args.save)

if __name__ == '__main__':
    main()