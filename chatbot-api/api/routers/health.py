from fastapi import APIRouter
router = APIRouter()
@router.get("/")
def get_answer():
    answer = "Success"
    print(answer)
    return answer

