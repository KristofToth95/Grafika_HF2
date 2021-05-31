//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Tóth Kristóf
// Neptun : E1GP69	
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const float epsilon = 0.0001f;
enum MaterialType {ROUGH, REFLECTIVE, DODE};
struct Material {
	vec3 ka, kd, ks;
	float shininess;
	vec3 F0;
	MaterialType type;
	Material( MaterialType t ){
		type = t;
	}
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one + kappa * kappa));
	}
};

struct ReflectiveMaterialDode : Material {
	ReflectiveMaterialDode(vec3 n, vec3 kappa) : Material(DODE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one + kappa * kappa));
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; };
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) {
		start = _start; dir = normalize(_dir);
	}
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);

	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x, eye.y, -(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
	float getZ() {
		return eye.z;
	}
};

struct Light {
	vec3 position;
	vec3 Le;
	Light(vec3 _position, vec3 _Le) {
		position = _position;
		Le = _Le;
	}
};
const float g = 0.618;
const float G = 1.618;

struct DodekaederShiny : public Intersectable {
	std::vector<vec3> vertices;
	std::vector<int> planes;
	float scale;
	int objFaces;

	DodekaederShiny(Material* _material, float _scale) {
		vertices = { vec3(0, g, G), vec3(0, -g, G), vec3(0, -g, -G), vec3(0, g, -G), vec3(G, 0, g),
					vec3(-G, 0, g), vec3(-G, 0, -g), vec3(G, 0, -g), vec3(g, G, 0), vec3(-g, G, 0),
					vec3(-g, -G, 0), vec3(g, -G, 0), vec3(1,1,1), vec3(-1, 1, 1), vec3(-1, -1, 1),
					vec3(1, -1, 1), vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1) };
		planes = { 1, 2, 16, 5, 13,  1, 13, 9, 10, 14,  1, 14, 6, 15, 2,  2, 15, 11, 12, 16,  3, 4, 18, 8, 17,  3, 17, 12, 11, 20,  3, 20, 7, 19, 4,
			19, 10, 9, 18, 4,  16, 12, 17, 8, 5,  5, 8, 18, 9, 13,  14, 10, 19, 7, 6,  6, 7, 20, 11, 15 };
		objFaces = 12;
		material = _material;
		scale = _scale;
	}
	void getObjPlane(int i, vec3* p, vec3* normal) {
		vec3 p1 = vertices[planes[5 * i] - 1], p2 = vertices[planes[5 * i + 1] - 1], p3 = vertices[planes[5 * i + 2] - 1];
		vec3 n = cross(p2 - p1, p3 - p1);
		if (dot(p1, n) < 0) *normal = -n;
		else *normal = n;
		*p = p1 * scale + vec3(0, 0, 0.03f);
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		for (int i = 0; i < objFaces; i++) {
			vec3 p1, normal;
			getObjPlane(i, &p1, &normal);
			float ti = fabs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
			if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;
			vec3 pintersect = ray.start + ray.dir * ti;
			bool outside = false;
			for (int j = 0; j < objFaces; j++) {
				if (i == j) continue;
				vec3 p11, n;
				getObjPlane(j, &p11, &n);
				if (dot(n, pintersect - p11) > 0) {
					outside = true;
					break;
				}
			}
			if (!outside) {				
					hit.t = ti;
					hit.position = pintersect;
					hit.normal = normalize(normal);
					hit.material = material;			
			}
		}
		return hit;
	}
	
};
struct DodekaederNotShiny : public Intersectable {
	std::vector<vec3> vertices;
	std::vector<int> planes;
	float scale;
	int objFaces;
	float holeSize;

	DodekaederNotShiny(Material* _material, float _scale, float _hole) {
		vertices = { vec3(0, g, G), vec3(0, -g, G), vec3(0, -g, -G), vec3(0, g, -G), vec3(G, 0, g),
					vec3(-G, 0, g), vec3(-G, 0, -g), vec3(G, 0, -g), vec3(g, G, 0), vec3(-g, G, 0), 
					vec3(-g, -G, 0), vec3(g, -G, 0), vec3(1,1,1), vec3(-1, 1, 1), vec3(-1, -1, 1), 
					vec3(1, -1, 1), vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1) };
		planes = { 1, 2, 16, 5, 13,  1, 13, 9, 10, 14,  1, 14, 6, 15, 2,  2, 15, 11, 12, 16,  3, 4, 18, 8, 17,  3, 17, 12, 11, 20,  3, 20, 7, 19, 4,  
			19, 10, 9, 18, 4,  16, 12, 17, 8, 5,  5, 8, 18, 9, 13,  14, 10, 19, 7, 6,  6, 7, 20, 11, 15 };
		objFaces = 12;
		material = _material;
		scale = _scale;
		holeSize = _hole;
	}
	void getObjPlane(int i, vec3 *p, vec3 *normal) {
		vec3 p1 = vertices[planes[5 * i] - 1], p2 = vertices[planes[5 * i + 1] - 1], p3 = vertices[planes[5 * i + 2] - 1];
		vec3 n = cross(p2 - p1, p3 - p1);
		if (dot(p1, n) < 0) *normal = -n;
		else *normal = n;
		*p = p1 * scale + vec3(0, 0, 0.03f);
	}
	bool getLineDistance(int i, vec3 p) {
		std::vector <vec3> points = { vertices[planes[5 * i] - 1], vertices[planes[5 * i + 1] - 1],  vertices[planes[5 * i + 2] - 1],  vertices[planes[5 * i + 3] - 1], vertices[planes[5 * i + 4] - 1] };
		float t1;
		for (int j = 0; j < 5; j++) {
			vec3 p1 = points[j];
			vec3 p2;
			if (j + 1 < 5)
				p2 = points[j + 1];
			else
				p2 = points[0];

			vec3 d = (p2-p1)/length(p2 - p1);
			vec3 unitV = p - p1;
			float t = dot(unitV, d);
			vec3 point = p1 + t*d;
			t1 = length(point - p);
			if ( t1 < holeSize)
				return true;
		}
		return false;
	}
	Hit intersect(const Ray &ray) {
		Hit hit;
		for (int i = 0; i < objFaces; i++) {
			vec3 p1, normal;
			getObjPlane(i, &p1, &normal);
			float ti = fabs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
			if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;
			vec3 pintersect = ray.start + ray.dir * ti;
			bool outside = false;
			for (int j = 0; j < objFaces; j++) {
				if (i == j) continue;
				vec3 p11, n;
				getObjPlane(j, &p11, &n);
				if (dot(n, pintersect - p11) > 0) {
					outside = true;
					break;
				}
			}
			if (!outside) {
				if (getLineDistance(i, pintersect))
				{
					hit.t = ti;
					hit.position = pintersect;
					hit.normal = normalize(normal);
					hit.material = material;
				}
			}
		}
		return hit;
	}
};
struct Agymenes : public Intersectable {
	mat4 Q;
	float r;
	vec3 translation;

	Agymenes(mat4& _Q, float _r, vec3 _translation, Material* _material) {
		Q = _Q;
		r = _r;
		translation = _translation;
		material = _material;
	}

	vec3 gradf(vec3 r) {
		vec4 g = vec4(r.x, r.y, r.z, 1) * Q * 2;
		return vec3(g.x, g.y, g.z);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 start = ray.start - translation;
		vec4 S(start.x, start.y, start.z, 1), D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		float a = dot(D * Q, D), b = dot(S * Q, D) * 2, c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;
		vec3 p1 = ray.start + ray.dir * t1;
		if (p1.x > r || p1.y > r || p1.z > r || p1.x < -r || p1.y < -r || p1.z < -r) t1 = -1;

		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
		if (p2.x > r || p2.y > r || p2.z > r || p2.x < -r || p2.y < -r || p2.z < -r) t2 = -1;

		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0) hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t2 < t1) hit.t = t2;
		else hit.t = t1;

		hit.position = start + ray.dir * hit.t;
		hit.normal = normalize(gradf(hit.position));
		hit.position = hit.position + translation;
		hit.material = material;
		return hit;
	}
};

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, 1), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.8f, 0.8f, 0.8f);
		vec3 lDir(1, 1, 0), Le(2, 2, 2);
		lights.push_back(new Light(lDir, Le));

		mat4 weirdsphere = mat4(60.5, 0, 0, 0,
								0, 50.5, 0, 0,
								0, 0, 0, 3.1,
								0, 0, 3.1, 0);
		objects.push_back(new Agymenes(weirdsphere, 0.3, vec3(0, 0, 0), new ReflectiveMaterial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9))));
		objects.push_back(new DodekaederNotShiny(new RoughMaterial(vec3(0.1f, 0.2f, 0.3f), vec3(2.0f, 2.0f, 2.0f), 50), 1.0f, 0.1f));
		objects.push_back(new DodekaederShiny(new ReflectiveMaterialDode(vec3(0, 0, 0), vec3(1, 1, 1)), 1 + epsilon));
	}
	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = (0, 0, 0);
		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			for (Light* light : lights) {
				vec3 lightDir = normalize(light->position - hit.position);
				float cosTheta = dot(hit.normal, lightDir);
				if (cosTheta > 0) {	
					vec3 LeIn = light->Le / dot(light->position - hit.position, light->position - hit.position);
					outRadiance = outRadiance + LeIn * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + lightDir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + LeIn * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}

		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}

		if (hit.material->type == DODE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			mat4 rotationM = RotationMatrix(72, hit.normal);
			vec4 newDir = vec4(reflectedDir.x, reflectedDir.y, reflectedDir.z, 1.0f);
			vec4 newDir1 = newDir * rotationM;
			vec4 newPos = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
			vec4 newPos1 = newPos * rotationM;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(vec3(newPos1.x, newPos1.y, newPos1.z) + hit.normal * epsilon, vec3(newDir1.x, newDir1.y, newDir1.z)), depth + 1) * F;
		}
		return outRadiance;
	}
	void Animate(float dt) {
		camera.Animate(dt);
	}
};
GPUProgram gpuProgram;
Scene scene;
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	out vec2 texcoord;	

	void main() {
		texcoord = (vp + vec2(1, 1))/2;
		gl_Position = vec4(vp.x, vp.y, 0, 1);		// transform vp from modeling space to normalized device space
	}
)";

const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform sampler2D textureUnit;
	in vec2 texcoord;
	out vec4 fragmentColor;	

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao, textureId = 0;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	void Load(std::vector<vec4>& image)
	{
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}
	void Draw() {
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
 };
FullScreenTexturedQuad* Quad;
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	Quad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
	
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	Quad->Load(image);
	Quad->Draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system

}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(0.1f);
	vec4 a = vec4(1, 2, 2, 1);
	vec4 b = vec4(4, 5, 2, 1);
	vec4 v = vec4(2, 5, 4, 4);
	float result = dot(a, b * v);
	printf("%f", result);
	glutPostRedisplay();
}
