#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "OLED.h"
#include "Servo.h"
#include "Encoder.h"

int16_t Num;
float Angle;

int main(void){
	OLED_Init();
	Servo_Init();
	Encoder_Init();
	
	OLED_ShowString( 1 , 1 , "Num:");
	OLED_ShowString( 2 , 1 , "Angle:");
	
	while (1){
		Num += Encoder_Get();
		OLED_ShowSignedNum( 1 , 5 , Num , 5);
		Angle = Num + 90;
		OLED_ShowSignedNum( 2 , 7 , (int16_t)Angle , 5);
		Servo_SetAngle(Angle);
	}
}