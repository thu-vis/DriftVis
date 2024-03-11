import os

from flask import Blueprint, render_template, session, send_file, request
from sklearn.svm import SVC

admin = Blueprint("admin", __name__)


@admin.route("/")
def index():
    # session["logged_in"] = True
    return render_template("index.html")


@admin.route("/svg")
def get_svg():
    path = os.path.join("svg", request.args.get('name'))
    return send_file(path)


# # following codes are for flask session test
# global_idx = 1
#
#
# @admin.route("/login/")
# def login():
#     global global_idx
#     # session["logged_in"] = True
#     # session["model"] = SVC(gamma=global_idx)
#     global_idx += 1
#     return "login" + str(global_idx)
#
#
# @admin.route("/test_login/")
# def test_login():
#     logged = session.get("logged_in")
#     model = session.get("model")
#     if logged:
#         # return "already logged in"
#         return str(model.get_params().get("gamma")) + "-model-params"
#     else:
#         return "not logged in "
#
#
# @admin.route("/logout/")
# def logout():
#     session.clear()
#     return "log out"
