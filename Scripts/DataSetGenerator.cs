using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public class DataSetGenerator : MonoBehaviour
{
    private float rotateX = 45;
    private int img_index = 0;
    private bool writeFinished = false;
    private List<GameObject> handObjectList = new List<GameObject>();
    private List<List<Vector3>> handJointGt = new List<List<Vector3>>();

    // Update is called once per frame
    void Start()
    {
        rotateX = transform.localEulerAngles.x + rotateX;
    }

    private void Update()
    {
        initHand();
        if (transform.localEulerAngles.x < rotateX)
        {
            transform.localRotation = Quaternion.Euler((float)(transform.localEulerAngles.x + 0.5), transform.localEulerAngles.y, transform.localEulerAngles.z);
            Camera.main.GetComponent<CameraController>().CameraCapture(Camera.main, "000" + img_index.ToString() + ".jpg");
            img_index += 1;
            handPosSave();
        }
        else
        {
            if (!writeFinished)
            {
                handJointGtSave("train_gt.txt");
                writeFinished = true;
            }
        }

    }

    private void initHand()
    {
        Transform[] myTransforms = GetComponentsInChildren<Transform>();
        foreach (var child in myTransforms)
            handObjectList.Add(child.gameObject);
    }

    private void handPosSave()
    {
        List<Vector3> handJointSingleGt = new List<Vector3>();
        foreach(var child in handObjectList)
        {
            handJointSingleGt.Add(Camera.main.WorldToScreenPoint(child.transform.position));
        }
        handJointGt.Add(handJointSingleGt);
    }

    private void handJointGtSave(string path)
    {
        string ouput_path = Application.dataPath + "/Output/" + path;
        string res = "";
        foreach(var childList in handJointGt)
        {
            foreach(var childVector in childList)
            {
                res += " " + childVector.ToString();
            }
            res += "\n";
        }

        File.WriteAllText(ouput_path, res, Encoding.Default);
        Debug.Log("train_gt Êä³öÍê±Ï");
    }
}
