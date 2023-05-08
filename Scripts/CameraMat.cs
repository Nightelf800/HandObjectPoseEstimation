using System.Collections;
using System.Collections.Generic;
using UnityEngine.XR.ARSubsystems;
using UnityEngine;
using System.IO;
using LitJson;

public class CameraMat : MonoBehaviour
{
    private GameObject camera_cur;
    private Dictionary<string, List<List<double>>> cameraParameter = new();
    private int dimension = 3;
    // Start is called before the first frame update
    void Start()
    {
        //CameraIntrinces();
        AddCameraParameterByFileStream();

    }

    /*private void CameraIntrinces()
    {
        List<List<double>> cameraIntrincesList = new();
        List<double> cameraIntrinces = new();
        Vector2 focallength = new Vector2(GetComponent<Camera>().focalLength, GetComponent<Camera>().fieldOfView);
        Vector2 principalPoint = new Vector2(UnityEngine.Screen.width / 2, UnityEngine.Screen.height / 2);

        cameraIntrinces.Add((focallength.x, 0, principalPoint.x));
        cameraIntrinces.Add(new Dimension(0, focallength.y, principalPoint.y));
        cameraIntrinces.Add(new Dimension(0, 0, 1));

        cameraParameter.Add("CamMat", cameraIntrinces);
        Debug.Log("相机内参：" + cameraIntrinces);
    }*/


    public void AddCameraParameterByFileStream()
    {
        string values = JsonMapper.ToJson(cameraParameter);
        string path = Application.dataPath + "/Output/CameraParameter.txt";
        // 文件流创建一个文本文件
        FileStream file = new FileStream(path, FileMode.Create);
        //得到字符串的UTF8 数据流
        byte[] bts = System.Text.Encoding.UTF8.GetBytes(values);
        // 文件写入数据流
        file.Write(bts, 0, bts.Length);
        if (file != null)
        {
            //清空缓存
            file.Flush();
            // 关闭流
            file.Close();
            //销毁资源
            file.Dispose();
        }
    }
}
