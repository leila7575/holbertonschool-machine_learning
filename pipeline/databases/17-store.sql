-- Creates trigger to decrease quantity of item based on orders of this item
DELIMITER //
CREATE TRIGGER quantity_update
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
	UPDATE items
	SET quantity = quantity - NEW.number
	WHERE items.name = NEW.item_name;
END//
DELIMITER ;
