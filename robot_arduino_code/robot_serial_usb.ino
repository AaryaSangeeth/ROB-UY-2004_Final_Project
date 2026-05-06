
// Laptop sends lines: speed,steering_angle\n
// steering_angle is offset from steering_angle_center 
// Laptop receives: TEL:encoder,steering_angle,num_lidar_rays[,angle,dist,...]\n


#include <Servo.h>
#include "RPLidar.h"
#include <string.h>

#define SendDeltaTimeInMs       100      
#define NoSignalDeltaTimeInMs   15000    

int last_time_rx = 0;
int last_time_tx = 0;

// Lidar setup 
#define RPLidarMotorPin         3
#define NumLidarRaysPerMsg      50
RPLidar lidar;
String current_lidar_scan_data;
int current_num_lidar_rays;

// Motor control setup 
#define RightSpeedPin           9
#define RightMotorDirPin1       12
#define RightMotorDirPin2       11
#define LeftSpeedPin            6
#define LeftMotorDirPin1        7
#define LeftMotorDirPin2        8

// Servo control setup 
#define ServoPin                10
Servo myServo;

// Encoder setup (
#define EncoderOutputA          4
#define EncoderOutputB          5
#define steering_angle_center   104       
int a_state;
int encoder_a_last_state;
int encoder_count;

// Structure for storing control signals received from laptop 
struct ControlSignal {
  int speed = 0;
  int steering_angle = 0;
};
ControlSignal last_control_signal;

// Structure for storing sensor signals sent to laptop 
struct SensorSignal {
  int encoder_count = 0;
  int steering_angle = 0;
  int num_lidar_rays = 0;
  String lidar_scan_data = "";
};
SensorSignal last_sensor_signal;

// USB: buffer one complete line 
char pending_usb_command[256];
volatile bool pending_usb_ready = false;
char usb_line_accum[256];
size_t usb_line_len = 0;

void setup()
{
  Serial.begin(115200);
  Serial.println("Running robot base code (USB serial)!");

  Serial2.begin(460800);
  lidar.begin(Serial2);
  delay(1000);
  if (lidar.begin(Serial2)) {
    Serial.println("Started Lidar!");
  } else {
    Serial.println("Failed Lidar!");
  }
  pinMode(RPLidarMotorPin, OUTPUT);
  reset_lidar_message();

  pinMode(RightMotorDirPin1, OUTPUT);
  pinMode(RightMotorDirPin2, OUTPUT);
  pinMode(LeftSpeedPin, OUTPUT);
  pinMode(LeftMotorDirPin1, OUTPUT);
  pinMode(LeftMotorDirPin2, OUTPUT);
  pinMode(RightSpeedPin, OUTPUT);
  stop();

  myServo.attach(ServoPin);
  myServo.write(steering_angle_center);

  pinMode(EncoderOutputA, INPUT);
  pinMode(EncoderOutputB, INPUT);

  last_time_rx = millis();
  last_time_tx = millis();
}

void loop()
{
  usb_accumulate_lines();

  ControlSignal control_signal = receive_control_signals(last_control_signal);
  last_control_signal = control_signal;

  control_robot(control_signal);

  SensorSignal sensor_signal = get_sensor_signal(control_signal.steering_angle);
  send_sensor_signal(sensor_signal);
}

// Read bytes from USB and stash full lines for receive_control_signals().
void usb_accumulate_lines()
{
  while (Serial.available()) {
    int c = Serial.read();
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      if (usb_line_len >= sizeof(usb_line_accum)) {
        usb_line_len = 0;
        continue;
      }
      usb_line_accum[usb_line_len] = '\0';
      strncpy(pending_usb_command, usb_line_accum, sizeof(pending_usb_command) - 1);
      pending_usb_command[sizeof(pending_usb_command) - 1] = '\0';
      pending_usb_ready = true;
      usb_line_len = 0;
      continue;
    }
    if (usb_line_len < sizeof(usb_line_accum) - 1) {
      usb_line_accum[usb_line_len++] = (char)c;
    } else {
      usb_line_len = 0;
    }
  }
}

void reset_lidar_message()
{
  current_num_lidar_rays = 0;
  current_lidar_scan_data = "";
}

void stop()
{
  digitalWrite(RightMotorDirPin1, LOW);
  digitalWrite(RightMotorDirPin2, LOW);
  digitalWrite(LeftMotorDirPin1, LOW);
  digitalWrite(LeftMotorDirPin2, LOW);
}

void forward(int speed)
{
  digitalWrite(RightMotorDirPin1, HIGH);
  digitalWrite(RightMotorDirPin2, LOW);
  digitalWrite(LeftMotorDirPin1, HIGH);
  digitalWrite(LeftMotorDirPin2, LOW);
  analogWrite(LeftSpeedPin, speed * 0.75);
  analogWrite(RightSpeedPin, speed);
}

ControlSignal receive_control_signals(ControlSignal last_control_signal_in)
{
  ControlSignal control_signal = last_control_signal_in;

  int new_time_rx = millis();
  if (pending_usb_ready) {
    char cmd_copy[256];
    strncpy(cmd_copy, pending_usb_command, sizeof(cmd_copy));
    cmd_copy[sizeof(cmd_copy) - 1] = '\0';
    pending_usb_ready = false;

    control_signal = unpack_control_signal(cmd_copy);
    Serial.print("Received cmd: ");
    Serial.print(control_signal.speed);
    Serial.print(", ");
    Serial.println(control_signal.steering_angle);
    last_time_rx = new_time_rx;
  }

  if (new_time_rx - last_time_rx > NoSignalDeltaTimeInMs) {
    control_signal.speed = 0;
    control_signal.steering_angle = 0;
  }
  return control_signal;
}

SensorSignal get_sensor_signal(float steering_angle)
{
  encoder_update();
  last_sensor_signal.steering_angle = steering_angle;
  last_sensor_signal.encoder_count = encoder_count;

  lidar_update();

  return last_sensor_signal;
}

void lidar_update()
{
  // Drain any USB bytes before blocking so partial lines advance; cannot interrupt waitPoint().
  usb_accumulate_lines();

  if (IS_OK(lidar.waitPoint())) {
    float distance = lidar.getCurrentPoint().distance;
    if (distance > 100 && current_num_lidar_rays < NumLidarRaysPerMsg) {
      int angle = int(lidar.getCurrentPoint().angle);
      current_num_lidar_rays += 1;
      current_lidar_scan_data += "," + String(angle) + "," + String(int(distance));
    }
  } else {
    analogWrite(RPLidarMotorPin, 255);

    rplidar_response_device_info_t info;
    if (IS_OK(lidar.getDeviceInfo(info, 100))) {
      lidar.startScan();
      analogWrite(RPLidarMotorPin, 255);
      delay(1000);
    }
  }
}

void encoder_update()
{
  a_state = digitalRead(EncoderOutputA);
  if (a_state != encoder_a_last_state) {
    if (digitalRead(EncoderOutputB) != a_state) {
      encoder_count++;
    } else {
      encoder_count--;
    }
  }
  encoder_a_last_state = a_state;
}

void send_sensor_signal(SensorSignal sensor_signal)
{
  int new_time_tx = millis();
  if (new_time_tx - last_time_tx > SendDeltaTimeInMs) {
    String msg = String(sensor_signal.encoder_count) + ",";
    msg = msg + String(sensor_signal.steering_angle) + ",";
    msg = msg + String(current_num_lidar_rays);
    msg = msg + current_lidar_scan_data;
    reset_lidar_message();

    Serial.print("TEL:");
    Serial.println(msg);

    last_sensor_signal.lidar_scan_data = "";
    last_sensor_signal.num_lidar_rays = 0;
    last_time_tx = new_time_tx;
  }
}

void control_robot(ControlSignal control_signal)
{
  forward(2 * control_signal.speed);

  int desired_angle = steering_angle_center + control_signal.steering_angle;
  myServo.write(desired_angle);
}

ControlSignal unpack_control_signal(char* packed_control_signal_as_char)
{
  ControlSignal control_signal;
  char* token;

  token = strtok(packed_control_signal_as_char, ",");
  control_signal.speed = atof(token);

  token = strtok(NULL, ",");
  control_signal.steering_angle = atof(token);

  return control_signal;
}
