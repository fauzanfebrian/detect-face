from urllib.request import Request, urlopen
import numpy as np
import cv2


def faceDetect(url):
    request_site = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    url_response = urlopen(request_site)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, -1)

    faceDetect = cv2.CascadeClassifier(
        "./models/lbpcascade_frontalface_improved.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = faceDetect.detectMultiScale(gray)

    if len(faces) < 1:
        return False

    return True


def faceDetects(urls, xml):
    result = []
    for url in urls:
        request_site = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        url_response = urlopen(request_site)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)

        faceDetect = cv2.CascadeClassifier(xml)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = faceDetect.detectMultiScale(gray)

        if len(faces) < 1:
            result.append(False)
            continue

        result.append(True)
    return result.count(True)


def faceDetectLive():
    camera = cv2.VideoCapture(0)
    faceDetect = cv2.CascadeClassifier(
        "./models/lbpcascade_frontalface_improved.xml"
    )

    while True:
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Face", img)
        if cv2.waitKey(1) == ord("q"):
            break


def test():
    urls = [
        "https://image.cnbcfm.com/api/v1/image/106467352-1585602933667virus-medical-flu-mask-health-protection-woman-young-outdoor-sick-pollution-protective-danger-face_t20_o07dbe.jpg?v=1585602987",
        "https://www.alphaindustries.eu/media/image/d8/f7/bd/128942-14-alpha-industries-label-ripstop-face-masks-001.jpg",
        "https://assets.burberry.com/is/image/Burberryltd/8F9D836A-AD7F-4072-8587-57C12E3957B3.jpg?$BBY_V2_ML_1X1$&wid=1920&hei=1920",
        "https://i0.wp.com/post.healthline.com/wp-content/uploads/2020/05/Female_Mask_Train_Station_1296x728-header.jpg?w=1155&h=1528",
        "https://images.indianexpress.com/2021/01/face-mask_1200_pixabay.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSv-J3DWMTXoKDEu46Ci6yffHV9PHo4XjcchA&usqp=CAU",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpOqRDFgM_Wn0hpY2n8ZRHRIQewkbCg84Upg&usqp=CAU",
    ]

    urlsFalse = [
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTW6509Xrnys2Pp_8RB3ffNOsqDrKlPtkvkRQ&usqp=CAU",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbZTPBifRk4nLz9dLqgKsJxFIkt8HgzGfqNQ&usqp=CAU",
        "https://img.rawpixel.com/private/static/images/website/2022-05/px142077-image-kwvvvktc.jpg?w=800&dpr=1&fit=default&crop=default&q=65&vib=3&con=3&usm=15&bg=F4F4F3&ixlib=js-2.2.1&s=0cfd8aa6e66ad55612f900d479a9d0de",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQLSd-cpTt3dT0HewjovjmJTDNlwGz4m7fyoA&usqp=CAU",
        "https://cdn.eso.org/images/thumb300y/eso1907a.jpg",
        "https://www.kibrispdr.org/data/663/images-of-flowers-0.jpg",
    ]

    print("Higher Score Better:")
    res = faceDetects(urls, "./models/lbpcascade_frontalface_improved.xml")
    print("lbpcascade_frontalface_improved:", res, len(urls))

    res = faceDetects(urls, "./models/haarcascade_frontalface_alt2.xml")
    print("haarcascade_frontalface_alt2:", res, len(urls))

    res = faceDetects(urls, "./models/haarcascade_frontalface_alt3.xml")
    print("haarcascade_frontalface_alt3:", res, len(urls))

    res = faceDetects(urls, "./models/haarcascade_eye.xml")
    print("haarcascade_eye:", res, len(urls))

    res = faceDetects(urls, "./models/haarcascade_profileface.xml")
    print("haarcascade_profileface:", res, len(urls))

    print("\nLower Score Better:")
    res = faceDetects(urlsFalse, "./models/lbpcascade_frontalface_improved.xml")
    print("lbpcascade_frontalface_improved:", res, len(urlsFalse))

    res = faceDetects(urlsFalse, "./models/haarcascade_frontalface_alt2.xml")
    print("haarcascade_frontalface_alt2:", res, len(urlsFalse))

    res = faceDetects(urlsFalse, "./models/haarcascade_frontalface_alt3.xml")
    print("haarcascade_frontalface_alt3:", res, len(urlsFalse))

    res = faceDetects(urlsFalse, "./models/haarcascade_eye.xml")
    print("haarcascade_eye:", res, len(urlsFalse))

    res = faceDetects(urlsFalse, "./models/haarcascade_profileface.xml")
    print("haarcascade_profileface:", res, len(urlsFalse))


if __name__ == "__main__":
    faceDetectLive()
    # test()

