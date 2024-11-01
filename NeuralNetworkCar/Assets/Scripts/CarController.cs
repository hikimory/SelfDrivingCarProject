using System;
using System.IO;
using System.Text;
using UnityEngine;

public class CarController : MonoBehaviour
{
    [SerializeField]
    private float _acceleration = 1.5f;
    [SerializeField]
    private float _maxSpeed = 6f;
    [SerializeField]
    private float _turnSpeed = 150f;
    [SerializeField]
    private float _brakeSpeed = 5f;

    [SerializeField]
    private Transform _lasersTransform;

    private float _currentSpeed = 0f;
    private NeuralNetwork.NeuralNetwork nn;
    private NeuralNetwork.ActivationFunctions.IActivationFunction funct;
    private Lasers _lasers;
    private string fileName;

    void Start()
    {
        nn = new NeuralNetwork.NeuralNetwork(new NeuralNetwork.ActivationFunctions.TanhActivationFunction(), new uint[] { 5, 3, 2 });
        nn.LoadWeightsAndBiases("Assets/Data_NN/Val_500_5.txt");
        funct = new NeuralNetwork.ActivationFunctions.SigmoidActivationFunction();
        _lasers = _lasersTransform.GetComponent<Lasers>();
        //fileName = $"Car-training-{DateTime.Now.ToString("HH_mm_ss")}.txt";
    }

    void FixedUpdate()
    {
        //var horizontal = Input.GetAxis("Horizontal");
        //var vertical = Input.GetAxis("Vertical");

        //if (vertical > 0)
        //{
        //    _currentSpeed += _acceleration * Time.deltaTime;
        //    _currentSpeed = Mathf.Clamp(_currentSpeed, 0, _maxSpeed);
        //}
        //else
        //{
        //    _currentSpeed -= _brakeSpeed * Time.deltaTime;
        //    _currentSpeed = Mathf.Max(_currentSpeed, 0);
        //}

        //transform.position += transform.right * _currentSpeed * Time.deltaTime;
        //transform.Rotate(0, horizontal * _turnSpeed * Time.deltaTime, 0, Space.World);
        //var inputs = _lasers.GetValues();
        //var output = new double[2] { vertical, horizontal };
        //SaveData(fileName, inputs, output);

        var inputs = _lasers.GetValues();
        var data = nn.FeedForward(inputs);
        var vertical = funct.CalculateOutput(data[0]);
        var horizontal = data[1];

        if (vertical > 0)
        {
            _currentSpeed += _acceleration * Time.deltaTime;
            _currentSpeed = Mathf.Clamp(_currentSpeed, 0, _maxSpeed);
        }
        else
        {
            _currentSpeed -= _brakeSpeed * Time.deltaTime;
            _currentSpeed = Mathf.Max(_currentSpeed, 0);
        }

        transform.position += transform.right * _currentSpeed * Time.deltaTime;
        transform.Rotate(0, (float)horizontal * _turnSpeed * Time.deltaTime, 0, Space.World);
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.gameObject.layer == LayerMask.NameToLayer("Wall"))
        {
            _currentSpeed = 0;
            GetComponent<Rigidbody>().isKinematic = true;
            this.enabled = false;
        }
    }

    private void SaveData(string filePath, double[] input, double[] output)
    {
        using (StreamWriter writer = new StreamWriter(filePath, true))
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(string.Join(" ", input));
            sb.Append(" ");
            sb.Append(string.Join(" ", output));

            writer.WriteLine(sb.ToString());
        }
    }
}
