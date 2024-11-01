using UnityEngine;

public class Lasers : MonoBehaviour
{
    [SerializeField]
    private int _rayLength = 15;

    [SerializeField]
    private int _rayCount = 5;

    [SerializeField]
    private float _angleOffset = 45f;

    [SerializeField]
    private LayerMask _raycastMask;

    private double[] inputs;

    void Awake()
    {
        inputs = new double[_rayCount];
    }

    void FixedUpdate()
    {
        for (int i = 0; i < _rayCount; i++)
        {
            float angle = i * _angleOffset;

            Quaternion rotation = Quaternion.AngleAxis(angle, transform.up);

            Vector3 direction = rotation * transform.forward;

            Vector3 origin = transform.position;

            RaycastHit hit;
            Ray Ray = new Ray(origin, direction);

            if (Physics.Raycast(Ray, out hit, _rayLength, _raycastMask))
            {
                inputs[i] = (_rayLength - hit.distance) / _rayLength;
                Debug.DrawLine(Ray.origin, hit.point, Color.red);
            }
            else
            {
                inputs[i] = 1;
            }
        }
    }

    public double[] GetValues()
    {
        return inputs;
    }
}
