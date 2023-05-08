using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using LitJson;

public class CameraController : MonoBehaviour
{
    private TestDataInfo testDataInfo = new();
    
    // Start is called before the first frame update
    void Start()
    {
        
        //testDataInfo.testDataGenerate(GetComponent<Camera>().focalLength, GetComponent<Camera>().fieldOfView);
    }

    public void CameraCapture(Camera m_Camera, string filename)
    {
        Debug.Log("ÕýÔÚäÖÈ¾Í¼Æ¬");
        RenderTexture rt = new RenderTexture(Screen.width, Screen.height, 16);
        m_Camera.targetTexture = rt;
        m_Camera.Render();
        RenderTexture.active = rt;
        Texture2D t = new Texture2D(Screen.width, Screen.height);
        t.ReadPixels(new Rect(0, 0, t.width, t.height), 0, 0);
        t.Apply();

        string path = Application.dataPath + "/Output/img_data/" + filename;
        System.IO.File.WriteAllBytes(path, t.EncodeToJPG());
        m_Camera.targetTexture = null;
    }


}
