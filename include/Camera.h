#pragma once
#include "MathBase.h"
#include <cmath> 
#include <Eigen/Dense>

class Camera {
public:
	static Matrix4f GetModelMatrix(float angle, const Vector3f& translation) {

		float rad = angle * M_PI / 180.0f;
		float cosA = std::cos(rad);
		float sinA = std::sin(rad);

		Matrix4f rotation;
		rotation(0, 0) = cosA;
		rotation(0, 2) = sinA;
		rotation(2, 0) = -sinA;
		rotation(2, 2) = cosA;

		Matrix4f trans = Matrix4f::Identity();

		trans(0, 3) = translation.x();
		trans(1, 3) = translation.y();
		trans(2, 3) = translation.z();

		return trans * rotation;
	}

	static Matrix4f GetViewMatrix(const Vector3f& eye, const Vector3f& target, const Vector3f& up) {
		Vector3f f = (target - eye).normalized();
		Vector3f u = up.normalized();
		Vector3f r = f.cross(u).normalized();
		u = r.cross(f);

		Matrix4f view = Matrix4f::Identity();
		view(0, 0) = r.x();
		view(0, 1) = r.y();
		view(0, 2) = r.z();
		view(1, 0) = u.x();
		view(1, 1) = u.y();
		view(1, 2) = u.z();
		view(2, 0) = -f.x();
		view(2, 1) = -f.y();
		view(2, 2) = -f.z();

		view(0, 3) = -r.dot(eye);
		view(1, 3) = -u.dot(eye);
		view(2, 3) = f.dot(eye);

		return view;
	}

	static Matrix4f GetProjectionMatrix(float fovY, float aspect, float zNear, float zFar) {
		Matrix4f proj = Matrix4f::Zero();

		float tanHalfFovY = std::tan(fovY * M_PI / 360.0f);
		proj(0, 0) = 1.0f / (aspect * tanHalfFovY);
		proj(1, 1) = 1.0f / tanHalfFovY;
		proj(2, 2) = -(zFar + zNear) / (zFar - zNear);
		proj(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);
		proj(3, 2) = -1.0f;

		return proj;
	}

	static Matrix4f ComputeMVP(float width, float height, int frameCount) {
		float angle = frameCount * 0.5f;
		Matrix4f model = GetModelMatrix(angle, Vector3f(0.0f, 0.0f, 0.0f));

		Vector3f eye(0.0f, 0.0f, 3.0f);
		Vector3f target(0.0f, 0.0f, 0.0f);
		Vector3f up(0.0f, 1.0f, 0.0f);
		Matrix4f view = GetViewMatrix(eye, target, up);

		float fovY = 45.0f;
		float aspect = width / height;
		float zNear = 0.1f;
		float zFar = 100.0f;

		Matrix4f projection = GetProjectionMatrix(fovY, aspect, zNear, zFar);

		return projection * view * model;
	}
};