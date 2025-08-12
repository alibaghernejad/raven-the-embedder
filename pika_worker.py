import logging
import pika
import json
from Settings import Settings
from embedding_pipeline import process_embedding_task
from retry import retry

settings = Settings()

def callback(ch, method, properties, body):
    event_json = body.decode()
    try:
        event = json.loads(event_json)
        task_id = event['Fields'][0]["TaskId"]
        process_embedding_task(task_id)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        try:
            logging.error(f"Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as inner_e:
            logging.error(f"Error acknowledging message: {inner_e}")

params = pika.URLParameters(settings.rabbitmq_broker_url)
@retry((pika.exceptions.AMQPConnectionError,pika.exceptions.ConnectionClosedByBroker), delay=5, max_delay=60, jitter=(1, 3))
def consume():

    logging.info(f"Connecting to RabbitMQ at {settings.rabbitmq_broker_url}")
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=settings.embedding_task_queue, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=settings.embedding_task_queue, on_message_callback=callback)
    logging.info(f"[*] Waiting for messages in queue '{settings.embedding_task_queue}'. To exit press CTRL+C")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
        connection.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    consume()
