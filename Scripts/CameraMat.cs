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
        Debug.Log("����ڲΣ�" + cameraIntrinces);
    }*/


    public void AddCameraParameterByFileStream()
    {
        string values = JsonMapper.ToJson(cameraParameter);
        string path = Application.dataPath + "/Output/CameraParameter.txt";
        // �ļ�������һ���ı��ļ�
        FileStream file = new FileStream(path, FileMode.Create);
        //�õ��ַ�����UTF8 ������
        byte[] bts = System.Text.Encoding.UTF8.GetBytes(values);
        // �ļ�д��������
        file.Write(bts, 0, bts.Length);
        if (file != null)
        {
            //��ջ���
            file.Flush();
            // �ر���
            file.Close();
            //������Դ
            file.Dispose();
        }
    }
}
