import open3d as o3d

pcd = o3d.io.read_point_cloud("face_3d_model.ply")
print(pcd)
o3d.visualization.draw_geometries([pcd])
