using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class JointConnectRender : MonoBehaviour
{
    public int cylinderResolution = 12;
    public float cylinderRadius = 0.1f;
    public Material cylinderMaterial;
    private bool castShadows = true;
    private Mesh cylinderMesh;
    private const float CYLINDER_MESH_RESOLUTION = 0.1f; //in centimeters, meshes within this resolution will be re-used
    private int curCylinderIndex = 0;
    private List<GameObject> JointList = new List<GameObject>();
    private Matrix4x4[] cylinderMatrices = new Matrix4x4[32];
    // Start is called before the first frame update
    void Start()
    {
        Transform[] myTransforms = GetComponentsInChildren<Transform>();
        foreach (var child in myTransforms)
            JointList.Add(child.gameObject);
    }

    // Update is called once per frame
    void Update()
    {
        curCylinderIndex = 0;

        //Draw cylinders between finger joints
        for (int i = 1; i < 21; i++)
        {
            int pre = 0;
            if (i % 4 != 1)
                pre = i - 1;
            else if (i > 1)
                pre = i - 4;

            Vector3 posA = JointList[i].transform.position;
            Vector3 posB = JointList[pre].transform.position;

            drawCylinder(posA, posB);
        }

        drawCylinder(JointList[0].transform.position, JointList[17].transform.position);

        // Draw Cylinders
        if (cylinderMesh == null) { cylinderMesh = getCylinderMesh(1f); }
        Graphics.DrawMeshInstanced(cylinderMesh, 0, cylinderMaterial, cylinderMatrices, curCylinderIndex, null,
          castShadows ? UnityEngine.Rendering.ShadowCastingMode.On : UnityEngine.Rendering.ShadowCastingMode.Off, true, gameObject.layer);
    }

    private bool isNaN(Vector3 v)
    {
        return float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z);
    }

    private void drawCylinder(Vector3 a, Vector3 b)
    {
        if (isNaN(a) || isNaN(b)) { return; }

        float length = (a - b).magnitude;

        if ((a - b).magnitude > 0.001f)
        {
            cylinderMatrices[curCylinderIndex++] = Matrix4x4.TRS(a,
              Quaternion.LookRotation(b - a), new Vector3(transform.lossyScale.x, transform.lossyScale.x, length));
        }
    }

    private Dictionary<int, Mesh> meshMap = new Dictionary<int, Mesh>();
    private Mesh getCylinderMesh(float length)
    {
        int lengthKey = Mathf.RoundToInt(length * 100 / CYLINDER_MESH_RESOLUTION);

        Mesh mesh;
        if (meshMap.TryGetValue(lengthKey, out mesh))
        {
            return mesh;
        }

        mesh = new Mesh();
        mesh.name = "GeneratedCylinder";
        mesh.hideFlags = HideFlags.DontSave;

        List<Vector3> verts = new List<Vector3>();
        List<Color> colors = new List<Color>();
        List<int> tris = new List<int>();

        Vector3 p0 = Vector3.zero;
        Vector3 p1 = Vector3.forward * length;
        for (int i = 0; i < cylinderResolution; i++)
        {
            float angle = (Mathf.PI * 2.0f * i) / cylinderResolution;
            float dx = cylinderRadius * Mathf.Cos(angle);
            float dy = cylinderRadius * Mathf.Sin(angle);

            Vector3 spoke = new Vector3(dx, dy, 0);

            verts.Add(p0 + spoke);
            verts.Add(p1 + spoke);

            colors.Add(Color.white);
            colors.Add(Color.white);

            int triStart = verts.Count;
            int triCap = cylinderResolution * 2;

            tris.Add((triStart + 0) % triCap);
            tris.Add((triStart + 2) % triCap);
            tris.Add((triStart + 1) % triCap);

            tris.Add((triStart + 2) % triCap);
            tris.Add((triStart + 3) % triCap);
            tris.Add((triStart + 1) % triCap);
        }

        mesh.SetVertices(verts);
        mesh.SetIndices(tris.ToArray(), MeshTopology.Triangles, 0);
        mesh.RecalculateBounds();
        mesh.RecalculateNormals();
        mesh.UploadMeshData(true);

        meshMap[lengthKey] = mesh;

        return mesh;
    }
}
